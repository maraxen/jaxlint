# jaxlint setup

One-shot onboarding command that exports skill files and registers the MCP server
across your tools.

## Usage

```bash
jaxlint setup                                           # Default: claude-code + gemini
jaxlint setup --tool cursor                             # Cursor only
jaxlint setup --tool claude-code --tool cursor          # Multiple surfaces
jaxlint setup --dry-run                                 # Preview without writing
jaxlint setup --no-skills                               # MCP registration only
jaxlint setup --no-mcp                                  # Skill export only
```

## Surfaces

| Surface | Skills | MCP registration |
|---------|--------|-----------------|
| `claude-code` | `~/.claude/skills/{name}/SKILL.md` | `claude mcp add jaxlint <cmd>` (falls back to `~/.claude.json`) |
| `gemini` | `~/.gemini/skills/{name}/SKILL.md` | `~/.gemini/settings.json` |
| `cursor` | — | `.cursor/mcp.json` in workspace root |

## Failure behaviour

Surface failures print a warning and continue; other surfaces are not affected.
Exit 0 if all surfaces succeeded, exit 1 if any failed.

## Idempotent

Re-running `jaxlint setup` is safe — skill files are overwritten with the current
version-stamped content; MCP config entries are merged (existing entries preserved).
