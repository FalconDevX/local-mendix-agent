# Maia MCP - manual test guide

Run this on the laptop where Mendix Studio Pro is installed. It covers acceptance tests 3 and 4,
which could not be executed on the build machine (no Studio Pro there). Expect about 15 minutes.

## 0. Prerequisites

- Mendix Studio Pro **11.10 or newer** (the MCP Server does not exist before 11.10).
- Studio Pro has internet access (Mendix requirement for the MCP Server).
- This app is installed (`setup.ps1`) and a chat model is downloaded (`/models`, `/download ...`).
  Sections 1-4 need **no model at all**.
- Use a **test app or a branch you can throw away** for section 5.

## 1. Enable the server in Studio Pro

1. Open an app in Studio Pro.
2. **Edit > Preferences > AI > MCP Server**, tick **Enable MCP Server**.
3. Note the **port** shown there (from 11.13 the active port is also in the status bar; if the
   configured port was taken Studio Pro picks another one).

## 2. Connect

```
.\run.bat
/mendix connect            # auto-discovery: ports of studiopro.exe, then candidate ports
/mendix connect 7782       # or give the port from Preferences explicitly (it is saved)
```

Expected:

```
  OK   Studio Pro process: pid ..., version 11.x.y, listening ports [...]
  OK   probe http://localhost:<port>/mcp: MCP initialize OK (...)
  OK   server: <name> <version>, protocol <date>
  OK   discovery: N tools, R resources, P prompts
Maia MCP connected at http://localhost:<port>/mcp
```

If it fails, every probe is listed with the reason. Common cases:

| Output | Meaning | Fix |
|---|---|---|
| `no running studiopro.exe found` | Studio Pro is not running (or runs under another user) | start it, open an app |
| `nothing is listening` on every candidate | server not enabled, or different port | section 1, then `/mendix connect <port>` |
| `port is open but MCP initialize failed` | something else owns that port, or a proxy intercepts localhost | check the port; `/doctor` must show `NO_PROXY: localhost,...` |
| `version ... is older than 11.10` | Studio Pro too old | upgrade |
| initialize OK but 0 tools | server running without an open app | open an app, `/mendix connect` again |

`/doctor` repeats these checks together with the network/proxy state.

## 3. Discover the real tools (acceptance test 3, part 1)

```
/mendix tools
/mendix capabilities
/mendix tools <one-tool-name>      # prints the JSON input schema
```

Check the **access** column. `read (annotation)` / `write (annotation)` come from the server.
`(name-heuristic)` or `(default)` mean the server sent no annotation and the app guessed
conservatively (unclear = write = asks first). If a read tool is shown as write (or the other way
round), fix it in `config/mcp_servers.yaml`:

```yaml
mendix:
  port: 7782
  tool_policy:
    some_read_tool: read
    some_dangerous_tool: deny
```

Please keep the output of `/mendix tools`: it is the authoritative tool list for your Studio Pro
version (the official docs do not publish one).

## 4. Read operation (acceptance test 3, part 2)

```
/mendix test
/mendix inspect
```

`/mendix test` must print `3. read call   OK <tool> ...` followed by real project data.
If it says "no argument-free read-only tool", call one by hand with arguments from its schema:

```
/mendix call <read-tool> {"<arg>": "<value>"}
```

Automated version:

```powershell
$env:LMA_MAIA_URL = "http://localhost:<port>/mcp"
.venv\Scripts\python -m pytest tests\test_maia_live.py -m maia -v -s
```

`test_connect_and_discover` and `test_read_operation` must pass.

## 5. Modification + verification (acceptance test 4)

Use a throw-away app/branch.

### 5a. Through the agent

```
/mendix
> Inspect this app and list its modules and the entities of the main module.
> In module <YourModule> add an entity "LmaTestTask" with a string attribute "Title" (200) and a
  DateTime attribute "DueDate". Then read the domain model back and check for errors.
```

Expected: a plan, `[CALLING MAIA MCP]` lines with the real tool names, a permission prompt before
each modifying tool (`y` / `a`), then a read-back and an error check. The run ends with an
**Operations** block listing what actually executed and whether it succeeded.

Verify in Studio Pro: the entity is visible in the domain model, the Errors pane is clean (or shows
exactly what the agent reported). Undo in Studio Pro with Ctrl+Z, or revert the branch.

### 5b. Deterministic (no model involved)

Pick a modifying tool and build its arguments from `/mendix tools <name>`, then:

```
/mendix call <write-tool> {...}
/mendix call <read-tool> {...}        # confirm the new state
```

or automated:

```powershell
$env:LMA_MAIA_WRITE_TOOL = "<write tool>"
$env:LMA_MAIA_WRITE_ARGS = '{"...": "..."}'
$env:LMA_MAIA_VERIFY_TOOL = "<read tool>"
$env:LMA_MAIA_VERIFY_ARGS = '{"...": "..."}'
$env:LMA_MAIA_VERIFY_CONTAINS = "LmaTestTask"
.venv\Scripts\python -m pytest tests\test_maia_live.py -m maia -v -s
```

## 6. Screenshots (acceptance test 8 with a real Studio Pro screenshot)

1. `/download qwen3-vl-4b` (or a larger vision model).
2. In Studio Pro press `Win+Shift+S` and capture a domain model or the Errors pane.
3. `/vision paste What entities and associations are on this screenshot?`

## 7. Offline check (acceptance test 10)

Disconnect the VPN/network. `/doctor` should show Hugging Face as unreachable and still run the
inference test. Chat, `/index`, `/search`, edits and `/memory` keep working. Whether Maia keeps
working offline depends on Studio Pro: Mendix states the MCP Server needs Studio Pro to be online.

## What to report back if something is off

- output of `/doctor` and `/mendix status`
- `data\logs\mcp-operations.jsonl` (every MCP call with arguments, duration, error)
- output of `/mendix tools`
