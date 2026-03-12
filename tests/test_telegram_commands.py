from nanobot.channels.telegram import TelegramChannel


def test_bot_commands_include_opencode_commands() -> None:
    commands = {command.command for command in TelegramChannel.BOT_COMMANDS}

    assert {"research", "status", "context"}.issubset(commands)
