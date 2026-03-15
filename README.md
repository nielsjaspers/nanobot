# Mike

Mike is the current bot implementation in this repository.

The old `nanobot/` package has been removed and replaced by the smaller Mike runtime.

Reference directories like `opencode/` and `beaker/` are kept only as local references and are not part of the active runtime.

## What Mike includes

- Telegram bot runtime
- OpenCode Go-backed models: `kimi-k2.5`, `minimax-m2.5`, `glm-5`
- Native local tool calling for normal chat work
- OpenCode-backed delegation and `/research`
- Shared `SOUL.md`, `USER.md`, `MEMORY.md`, and structured history archive
- Skills support

## Quick start

```bash
uv run mike onboard
```

Then edit:

- `~/.mike/config.json`

Set at least:

- `telegram_token`
- `opencode_api_key`

Start the bot:

```bash
uv run mike gateway
```

The first `gateway` or `agent` run auto-creates `~/.mike/config.json` if it does not exist yet.

Direct CLI message:

```bash
uv run mike agent -m "Hello"
```

## Commands

- `/new`
- `/clear`
- `/stop`
- `/restart`
- `/help`
- `/model`
- `/research`
- `/status`
- `/context`

Model aliases:

- `/model kimi`
- `/model minimax`
- `/model glm`

## Data location

Mike stores runtime data in:

- `~/.mike/`

Important files:

- `~/.mike/SOUL.md`
- `~/.mike/USER.md`
- `~/.mike/MEMORY.md`
- `~/.mike/history/index.json`
- `~/.mike/history/records/*.json`
- `~/.mike/sessions/*/active.json`
