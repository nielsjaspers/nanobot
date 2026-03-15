# Security Policy

## Reporting a Vulnerability

If you discover a security issue in Mike, please report it privately to the repository maintainers.

Do not open a public issue with exploit details.

Include:

- a short description
- reproduction steps
- impact
- suggested fix, if you have one

## Security Notes For Mike

- Store secrets in `~/.mike/config.json`
- Restrict that file with `chmod 600 ~/.mike/config.json`
- Keep `telegram_allow_from` configured for production use
- Review shell/file tool usage carefully
- Run Mike with a dedicated user account when deployed on a VPS
- Protect `~/.mike/` because it contains chat archives, memory, logs, and local state

## Recommended Permissions

```bash
chmod 700 ~/.mike
chmod 600 ~/.mike/config.json
```

## Incident Response

If you suspect compromise:

1. revoke API keys
2. rotate Telegram/OpenCode credentials
3. inspect `~/.mike/logs/`
4. inspect `~/.mike/history/` and `~/.mike/tasks/`
5. redeploy with fresh secrets
