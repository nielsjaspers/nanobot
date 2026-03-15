"""Telegram channel for Mike."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from loguru import logger
from telegram import BotCommand, ReplyParameters, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.request import HTTPXRequest

from mike.bus import MessageBus
from mike.common import split_message
from mike.config import MikeConfig
from mike.storage.chats import ChatStore
from mike.telegram.render import markdown_to_telegram_html
from mike.types import OutboundMessage

TELEGRAM_MAX_MESSAGE_LEN = 4000


class TelegramBot:
    BOT_COMMANDS = [
        BotCommand("start", "Start the bot"),
        BotCommand("new", "Start a new conversation"),
        BotCommand("clear", "Clear chat instantly"),
        BotCommand("stop", "Stop the current task"),
        BotCommand("help", "Show available commands"),
        BotCommand("restart", "Restart the bot"),
        BotCommand("model", "Show or switch model"),
        BotCommand("research", "Run a complex research task"),
        BotCommand("status", "Show running background tasks"),
        BotCommand("context", "Add context to a running task"),
    ]

    def __init__(self, config: MikeConfig, bus: MessageBus, store: ChatStore):
        self.config = config
        self.bus = bus
        self.store = store
        self._app: Application | None = None
        self._running = False
        self._typing_tasks: dict[str, asyncio.Task] = {}
        self._message_threads: dict[tuple[str, int], int] = {}

    def is_allowed(self, sender_id: str) -> bool:
        allow_list = self.config.telegram_allow_from
        if not allow_list:
            logger.warning("telegram: telegram_allow_from is empty - all access denied")
            return False
        if "*" in allow_list:
            return True
        if sender_id in allow_list:
            return True
        if sender_id.count("|") == 1:
            sid, username = sender_id.split("|", 1)
            return sid in allow_list or username in allow_list
        return False

    async def start(self) -> None:
        if not self.config.telegram_token:
            logger.error("Telegram bot token not configured")
            return
        self._running = True
        req = HTTPXRequest(
            connection_pool_size=16,
            pool_timeout=5.0,
            connect_timeout=30.0,
            read_timeout=30.0,
            proxy=self.config.telegram_proxy if self.config.telegram_proxy else None,
        )
        self._app = (
            Application.builder()
            .token(self.config.telegram_token)
            .request(req)
            .get_updates_request(req)
            .build()
        )
        self._app.add_error_handler(self._on_error)
        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("help", self._on_help))
        for name in ("new", "clear", "stop", "restart", "model", "research", "status", "context"):
            self._app.add_handler(CommandHandler(name, self._forward_command))
        self._app.add_handler(
            MessageHandler(
                (filters.TEXT | filters.PHOTO | filters.Document.ALL) & ~filters.COMMAND,
                self._on_message,
            )
        )
        await self._app.initialize()
        await self._app.start()
        bot_info = await self._app.bot.get_me()
        logger.info("Mike Telegram bot @{} connected", bot_info.username)
        try:
            await self._app.bot.set_my_commands(self.BOT_COMMANDS)
        except Exception as exc:
            logger.warning("Failed to register bot commands: {}", exc)
        await self._app.updater.start_polling(
            allowed_updates=["message"], drop_pending_updates=True
        )
        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        for chat_id in list(self._typing_tasks):
            self._stop_typing(chat_id)
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None

    async def send(self, msg: OutboundMessage) -> None:
        if not self._app:
            return
        if not msg.metadata.get("_progress", False):
            self._stop_typing(msg.chat_id)
        try:
            chat_id = int(msg.chat_id)
        except ValueError:
            logger.error("Invalid chat_id: {}", msg.chat_id)
            return
        reply_to_message_id = msg.metadata.get("message_id")
        message_thread_id = msg.metadata.get("message_thread_id")
        if message_thread_id is None and reply_to_message_id is not None:
            message_thread_id = self._message_threads.get((msg.chat_id, reply_to_message_id))
        thread_kwargs = (
            {"message_thread_id": message_thread_id} if message_thread_id is not None else {}
        )
        reply_params = None
        if self.config.telegram_reply_to_message and reply_to_message_id:
            reply_params = ReplyParameters(
                message_id=reply_to_message_id, allow_sending_without_reply=True
            )
        for media_path in msg.media or []:
            await self._send_media(chat_id, media_path, reply_params, thread_kwargs)
        if msg.content and msg.content != "[empty message]":
            is_progress = msg.metadata.get("_progress", False)
            for chunk in split_message(msg.content, TELEGRAM_MAX_MESSAGE_LEN):
                if not is_progress:
                    await self._send_with_streaming(chat_id, chunk, reply_params, thread_kwargs)
                else:
                    await self._send_text(chat_id, chunk, reply_params, thread_kwargs)

    async def bridge_outbound(self) -> None:
        while self._running:
            msg = await self.bus.consume_outbound()
            if msg.channel == "telegram":
                await self.send(msg)

    async def _forward_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_user:
            return
        user = update.effective_user
        sender = self._sender_id(user)
        if not self.is_allowed(sender):
            return
        message = update.message
        text = message.text or ""
        metadata = self._build_message_metadata(message, user)
        session_key = self._derive_topic_session_key(message)
        self._start_typing(str(message.chat_id))
        await self.bus.publish_inbound(
            self._build_inbound(sender, str(message.chat_id), text, [], metadata, session_key)
        )

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_user:
            return
        user = update.effective_user
        sender = self._sender_id(user)
        if not self.is_allowed(sender):
            return
        message = update.message
        if message.chat.type != "private" and self.config.telegram_group_policy == "mention":
            if not self._should_respond_to_group_message(message):
                return
        content = message.text or message.caption or ""
        media, content_parts = await self._download_message_media(message)
        if content_parts:
            if content:
                content = "\n".join(content_parts + [content])
            else:
                content = "\n".join(content_parts)
        metadata = self._build_message_metadata(message, user)
        session_key = self._derive_topic_session_key(message)
        self._start_typing(str(message.chat_id))
        await self.bus.publish_inbound(
            self._build_inbound(sender, str(message.chat_id), content, media, metadata, session_key)
        )

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not update.effective_user:
            return
        user = update.effective_user
        await update.message.reply_text(
            f"Hi {user.first_name}! I'm Mike.\n\nSend me a message and I'll respond.\nType /help to see available commands."
        )

    async def _on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return
        await update.message.reply_text(
            "Mike commands:\n"
            "/new - Start a new conversation\n"
            "/clear - Clear chat instantly\n"
            "/stop - Stop the current task\n"
            "/model - Show or switch model\n"
            "/research - Run a complex task in OpenCode\n"
            "/status - Show running background tasks\n"
            "/context - Add context to a running task\n"
            "/restart - Restart the bot\n"
            "/help - Show available commands"
        )

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.exception("Telegram error", exc_info=context.error)

    def _build_inbound(
        self,
        sender: str,
        chat_id: str,
        content: str,
        media: list[str],
        metadata: dict,
        session_key: str | None,
    ):
        from mike.types import InboundMessage

        return InboundMessage(
            channel="telegram",
            sender_id=sender,
            chat_id=chat_id,
            content=content,
            media=media,
            metadata=metadata,
            session_key_override=session_key,
        )

    def _sender_id(self, user) -> str:
        sid = str(user.id)
        return f"{sid}|{user.username}" if user.username else sid

    def _derive_topic_session_key(self, message) -> str | None:
        message_thread_id = getattr(message, "message_thread_id", None)
        if message.chat.type == "private" or message_thread_id is None:
            return None
        return f"telegram:{message.chat_id}:topic:{message_thread_id}"

    def _build_message_metadata(self, message, user) -> dict:
        reply_to = getattr(message, "reply_to_message", None)
        return {
            "message_id": message.message_id,
            "user_id": user.id,
            "username": user.username,
            "first_name": user.first_name,
            "is_group": message.chat.type != "private",
            "message_thread_id": getattr(message, "message_thread_id", None),
            "reply_to_message_id": getattr(reply_to, "message_id", None) if reply_to else None,
        }

    async def _download_message_media(self, msg) -> tuple[list[str], list[str]]:
        if not self._app:
            return [], []
        media_file = None
        media_type = None
        if getattr(msg, "photo", None):
            media_file = msg.photo[-1]
            media_type = "image"
        elif getattr(msg, "document", None):
            media_file = msg.document
            media_type = "file"
        if not media_file:
            return [], []
        try:
            file = await self._app.bot.get_file(media_file.file_id)
            ext = self._get_extension(
                media_type,
                getattr(media_file, "mime_type", None),
                getattr(media_file, "file_name", None),
            )
            session_key = self._derive_topic_session_key(msg) or f"telegram:{msg.chat_id}"
            path_str = self.store.save_upload(
                session_key, f"{media_file.file_id[:16]}{ext}", await file.download_as_bytearray()
            )
            return [path_str], [f"[{media_type}: {path_str}]"]
        except Exception as exc:
            logger.warning("Failed to download message media: {}", exc)
            return [], []

    @staticmethod
    def _get_extension(media_type: str, mime_type: str | None, file_name: str | None) -> str:
        if file_name and "." in file_name:
            return "." + file_name.rsplit(".", 1)[-1].lower()
        if media_type == "image":
            if mime_type == "image/jpeg":
                return ".jpg"
            if mime_type == "image/webp":
                return ".webp"
            return ".png"
        return ".bin"

    def _should_respond_to_group_message(self, message) -> bool:
        text = (message.text or message.caption or "").lower()
        bot_username = (
            self._app.bot.username.lower() if self._app and self._app.bot.username else ""
        )
        if bot_username and f"@{bot_username}" in text:
            return True
        reply = getattr(message, "reply_to_message", None)
        if (
            reply
            and getattr(reply, "from_user", None)
            and self._app
            and reply.from_user.id == self._app.bot.id
        ):
            return True
        return False

    def _start_typing(self, chat_id: str) -> None:
        if not self._app or chat_id in self._typing_tasks:
            return

        async def loop() -> None:
            while True:
                try:
                    await self._app.bot.send_chat_action(chat_id=int(chat_id), action="typing")
                    await asyncio.sleep(4)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    return

        self._typing_tasks[chat_id] = asyncio.create_task(loop())

    def _stop_typing(self, chat_id: str) -> None:
        task = self._typing_tasks.pop(chat_id, None)
        if task:
            task.cancel()

    async def _send_media(
        self, chat_id: int, media_path: str, reply_params, thread_kwargs: dict
    ) -> None:
        if not self._app:
            return
        ext = Path(media_path).suffix.lower()
        sender = self._app.bot.send_document
        param = "document"
        if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
            sender = self._app.bot.send_photo
            param = "photo"
        with open(media_path, "rb") as handle:
            await sender(
                chat_id=chat_id, **{param: handle}, reply_parameters=reply_params, **thread_kwargs
            )

    async def _send_text(
        self, chat_id: int, text: str, reply_params=None, thread_kwargs: dict | None = None
    ) -> None:
        if not self._app:
            return
        try:
            html = markdown_to_telegram_html(text)
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=html,
                parse_mode="HTML",
                reply_parameters=reply_params,
                **(thread_kwargs or {}),
            )
        except Exception as exc:
            logger.warning("HTML parse failed, falling back to plain text: {}", exc)
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_parameters=reply_params,
                **(thread_kwargs or {}),
            )

    async def _send_with_streaming(
        self, chat_id: int, text: str, reply_params=None, thread_kwargs: dict | None = None
    ) -> None:
        if not self._app:
            return
        draft_id = int(time.time() * 1000) % (2**31)
        try:
            step = max(len(text) // 8, 40)
            for idx in range(step, len(text), step):
                await self._app.bot.send_message_draft(
                    chat_id=chat_id, draft_id=draft_id, text=text[:idx]
                )
                await asyncio.sleep(0.04)
            await self._app.bot.send_message_draft(chat_id=chat_id, draft_id=draft_id, text=text)
            await asyncio.sleep(0.15)
        except Exception:
            pass
        await self._send_text(chat_id, text, reply_params, thread_kwargs)
