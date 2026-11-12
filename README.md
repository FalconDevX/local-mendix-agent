# Local Mendix Agent

A terminal AI coding agent in the spirit of Claude Code that runs **entirely on one Windows laptop**:
local GGUF models through llama.cpp, a multi-server MCP client with a dedicated integration for the
official **Mendix Studio Pro MCP server (Maia)**, project indexing, staged edits with diff/apply/undo,
persistent memory and screenshot understanding.

No Ollama, no Docker, no administrator rights, no cloud inference. After the initial downloads it
works offline.

## Status: what was verified and what was not

Built and tested on Windows 11, Python 3.12, RTX 4060 Laptop (8 GB VRAM), 2026-11-12.

| Acceptance test | Status | Evidence |
|---|---|---|
| 1. Local inference without Ollama/admin | **verified** | `tests/test_agent.py::test_local_model_inference_and_embeddings`, `/doctor` inference test |
| 2. MCP: connect, discover, execute a real tool | **verified** | `tests/test_mcp.py` against a real MCP server process (stdio + Streamable HTTP) |
| 3. Maia: connect to Studio Pro MCP, read operation | **not verified - no Studio Pro on the build machine** | implemented; run `tests/test_maia_live.py` and `docs/MAIA_TEST_GUIDE.md` on the target laptop |
| 4. Maia modification + verification | **not verified** (same reason) | same guide, section 5 |
| 5. Index a Python repo, multi-file questions | **verified** | `tests/test_core.py`, manual run in `docs/TEST_REPORT.md` |
| 6. Multi-file edit, unified diff, apply, verify | **verified** | `tests/test_core.py::test_multi_file_edit_diff_apply_undo` |
| 7. Memory survives restart, projects isolated | **verified** | `tests/test_core.py::test_memory_persists_and_isolates_projects` |
| 8. Vision on a Studio Pro screenshot | **verified on a synthetic image only** | local vision model read a drawn domain model + error box correctly; no real Studio Pro capture was available |
| 9. Autonomous filesystem + MCP workflow | **verified** | real 4B model run (file + MCP write + read-back) and `tests/test_agent.py` |
| 10. Offline operation | **verified (simulated)** | external network blocked for the process; no physical disconnect was done |

Details, numbers and the list of what is not covered: `docs/TEST_REPORT.md`.

The Maia integration is the top priority of this project and it is the one part that could not be
exercised end to end here. What exists is a real MCP client (official SDK) that was proven against a
real MCP server, plus Mendix-specific endpoint discovery, diagnostics and live tests that run as soon
as Studio Pro is available. **Nothing about Maia's tools is hardcoded**: names, schemas and
read/write classification come from the server at runtime.

## Install (no admin)

```powershell
# in the project folder
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\run.bat
```

`setup.ps1` creates `.venv` and installs the dependencies from `requirements.txt` (all have official
Windows wheels). Options for restricted networks:

```powershell
.\setup.ps1 -IndexUrl https://nexus.company/repository/pypi/simple     # corporate PyPI mirror
.\setup.ps1 -Wheelhouse .\wheelhouse                                    # fully offline
# build the wheelhouse on any machine with internet:
pip download -r requirements.txt -d wheelhouse --only-binary=:all: --python-version 3.12 --platform win_amd64
```

First start:

```
/doctor                 full diagnostics (Python, packages, GPU, disk, proxy, Hugging Face, MCP, Maia)
/models                 catalogue + what fits this hardware
/download runtime       official llama.cpp build (cpu | vulkan | cuda-12.4), unzipped into data\runtime
/download qwen3-8b      a chat model (asks for confirmation, resumable, SHA-256 verified)
/download nomic-embed-v1.5      embeddings for semantic search
/download qwen3-vl-4b   vision model for screenshots
/open C:\Projects\MyMendixApp
/index
/mendix connect         or: /mendix connect 7782
```

### If downloads are blocked

Nothing is bypassed and there is no cloud fallback. The app tells you what failed and you can supply
the files through an approved channel:

- runtime: download `llama-<tag>-bin-win-<variant>-x64.zip` from the llama.cpp GitHub releases, then
  `/download runtime C:\path\llama-...zip` (for CUDA also pass `cudart-llama-bin-win-cuda-12.4-x64.zip`),
  or set `LMA_LLAMA_SERVER` to an approved `llama-server.exe`.
- models: copy a `.gguf` file anywhere and `/models add C:\models\model.gguf chat`
  (vision: `/models add model.gguf vision mmproj.gguf`).
- proxy: set `HTTPS_PROXY`; corporate root CAs are taken from the Windows trust store.
- Hugging Face mirror: `/config llm.hf_endpoint https://hf-mirror.company`.

If `llama-server.exe` is blocked by application control (AppLocker/WDAC), the error says so; that
needs an IT exception for the file. There is no workaround in this app, by design.

## Mendix Studio Pro (Maia MCP)

Facts from the official documentation (docs.mendix.com/refguide/studio-pro-mcp-server, checked 2026-11-12):

- available from **Studio Pro 11.10**; enable under **Preferences > AI > MCP Server**, port configured there
- endpoint `http://localhost:<port>/mcp`, HTTP (Streamable HTTP) transport, no token for the local endpoint
- runs locally, no Studio Pro sign-in required, but Studio Pro needs an internet connection
- works on the app currently open in Studio Pro; changes appear live in the editors and the Errors pane
- 11.10 limitations: cannot generate pages; web-fetch tool calls wait for approval inside Studio Pro
- the docs list **no tool names and no default port**

Because of the last point the agent discovers everything:

1. `mendix.url` / `mendix.port` from `config/mcp_servers.yaml` if set,
2. otherwise the loopback ports the running `studiopro.exe` listens on,
3. otherwise `candidate_ports` (7782, reported by the mendixlabs Kiro power for 11.13/11.14 - not official).

Each candidate gets a real MCP `initialize`; then tools, schemas, resources and prompts are listed.
Tools are classified read/write from the server's annotations; without annotations a name heuristic
is used and anything unclear counts as **write** (asks before running). Correct it per tool with
`mendix.tool_policy`.

```
/mendix status         process, version, probes, connection, health
/mendix connect [port|url]
/mendix tools [name]   discovered tools, or the JSON schema of one
/mendix capabilities   what this Studio Pro version can read / modify
/mendix inspect        calls every argument-free read tool
/mendix test           connect + discover + one real read call
/mendix call <tool> <json>   manual call (modifying tools ask first)
/mendix <request>      run the Mendix agent, e.g. /mendix add an entity Task with Title and DueDate
```

In `/mendix` mode the system prompt is generated from the live tool list; the agent inspects first,
proposes, applies changes through the modifying tools (each needs approval, or "always" for the
session), re-reads and runs the server's error-check tool if one exists. If the server has no tool
for something, the agent is instructed to say so.

Known limits: page generation depends on the Studio Pro version; Java actions and custom widgets are
not reachable through the MCP server (community reports); `.mpr`/`.mxunit` model files are binary
and are never edited directly by this app.

## Modes and commands

Modes: `/ask` (read-only Q&A), `/plan` (plan only), `/agent` (autonomous), `/code` (edits),
`/review` (review git diff / staged changes), `/mendix`, `/vision`. `/ask what does X do` runs one
message in that mode without switching.

| | |
|---|---|
| Models | `/model`, `/models`, `/download` |
| Project | `/project`, `/open`, `/index [--full]`, `/search`, `/read` |
| Editing | `/write`, `/edit`, `/diff`, `/apply`, `/reject`, `/undo`, `/history` |
| MCP | `/mcp list|add|connect|disconnect|tools|inspect|test|call`, `/mendix ...` |
| Memory | `/memory [search|project|global|add|forget|clear-session]`, `/context` |
| Images | `/attach <file>|paste`, `/vision [file|paste] [question]`, `@screenshot.png` in a message |
| Other | `/config`, `/doctor`, `/clear`, `/help`, `/exit` |

Keys: `Alt+Enter` new line, `Tab` completion (commands, sub-commands, model ids, paths, `@file`),
`Ctrl+C` cancels the running task, `Ctrl+D` exits. Non-interactive use:

```powershell
.\run.bat -c "/doctor"
.\run.bat --project C:\repo -p "where is authentication implemented?" -m ask
.\run.bat --project C:\repo -p "rename foo to bar everywhere" -m code --yes    # --yes approves and applies
```

## How it works

```
main.py -> lma/cli.py
lma/config.py            settings, paths, secret references, proxy/trust-store setup
lma/llm/                 hardware.py  registry.py (catalogue, installed models)  downloader.py (resume + SHA-256)
                         runtime.py (llama.cpp install)  manager.py (llama-server processes, chat/vision/embeddings)
lma/mcpx/                manager.py (multi-server MCP client)  mendix.py (Maia discovery + diagnostics)
lma/core/                agent.py (loop)  context.py (prompts, retrieval, compression)  permissions.py  state.py
lma/tools/               toolbox.py (all agent tools + executor)  git.py  image.py
lma/project/             scanner.py (file kinds, Mendix detection)  indexer.py (chunks, FTS5, embeddings, symbols, deps)
lma/editing/editor.py    staged changes, diff, transactional apply, snapshots, undo
lma/memory/store.py      SQLite sessions, messages, project/global memories, hybrid retrieval
lma/ui/                  terminal.py (REPL)  commands.py (slash commands)  rendering.py
lma/doctor.py            /doctor
config/                  models.yaml  mcp_servers.yaml  settings.yaml(.example)
tests/                   pytest suite; fixtures/demo_server.py is a real MCP server for tests
```

Deliberate deviations from the original brief:

- **`llama-server.exe` instead of `llama-cpp-python`.** The Python binding needs a C++/CUDA
  toolchain for GPU builds on Windows; the official release ZIP needs nothing. The agent talks to it
  over `127.0.0.1` only.
- **Package is `lma/`, MCP code is `lma/mcpx/`.** A top-level `mcp/` folder would shadow the official
  `mcp` SDK.
- **Rich + prompt_toolkit instead of Textual.** A scrolling REPL behaves like Claude Code and works
  in every Windows terminal, including restricted ones.
- **Direct HTTPS downloads instead of the `huggingface_hub` package** (one dependency less; same
  Hub, resumable, verified against the Hub's LFS SHA-256).
- **File watcher is polling-based** (`index.watch_interval_s`), no native dependency.

### Agent loop

1. retrieve relevant memory, earlier-session summaries and code snippets (never the whole repo)
2. in `/agent`, `/code`, `/mendix`: ask the model for a numbered plan and show it
3. stream the model answer; execute requested tools through the permission layer
4. feed real tool results (or real errors) back; identical repeated calls are blocked
5. when the model stops, syntax-check staged files and send problems back once
6. print an **operations report built from the execution ledger**, not from the model's text
7. show staged diffs and ask to apply

Limits: `agent.max_steps`, per-tool timeout, model-call retries, context compression (old tool
output is shortened first, then older turns are summarised).

### Safety model

- file tools are confined to the open project folder (resolved paths, no `..` escapes)
- `write_file`/`edit_file`/`delete_file`/`rename_file` only **stage**; `/apply` snapshots originals,
  detects on-disk conflicts, rolls everything back if one write fails; `/undo` restores
- `/ask`, `/plan`, `/review`, `/vision` never get modifying tools
- MCP tools that are not clearly read-only, `create_dir`, `git_commit`, `run_command` ask every time
  (`a` = allow for this session). `run_command` is off unless `agent.allow_shell: true`
- Maia is only ever contacted on loopback addresses; no ports are opened to the network
- secrets in config are references (`keyring:NAME` -> Windows Credential Manager, or `env:NAME`)
- no telemetry, no external calls except explicit `/download` and `/doctor` connectivity probes

### Models

`config/models.yaml` is the catalogue; every repo/file was checked on the Hugging Face API. Size and
SHA-256 are re-read from the Hub at download time and verified. Tool calling: Qwen2.5-Coder and
Qwen3 chat templates declare tools and llama-server converts them to structured calls; small models
often write calls as text, which the agent also parses - but **models under ~4B parameters are not
reliable for multi-step agent work** (measured here with Qwen2.5-Coder-1.5B). Use `qwen3-8b` or
`qwen2.5-coder-7b` or larger for `/agent` and `/mendix`.

### Offline

Works without network: chat, agent, tools, indexing, search, edits, memory, vision, MCP on localhost.
Needs network: `/download`. Note that Studio Pro's MCP server itself requires Studio Pro to be online
(Mendix requirement, outside this app).

## Tests

```powershell
.venv\Scripts\python -m pytest -m "not maia and not model"     # no model, no Studio Pro needed
.venv\Scripts\python -m pytest -m model                        # needs runtime + a chat model
.venv\Scripts\python -m pytest -m maia -v -s                   # needs Studio Pro with MCP Server enabled
```

## Troubleshooting

| Symptom | What to do |
|---|---|
| `Maia MCP is not connected` | `/mendix status` lists every probe. Start Studio Pro 11.10+, open an app, enable the MCP Server, `/mendix connect <port>` |
| port open but `MCP initialize failed` | another program uses that port; set the right `mendix.port` |
| downloads fail with TLS/proxy errors | `/doctor` Network section; set `HTTPS_PROXY`; use a mirror or copy files manually |
| `llama-server exited` / out of memory | smaller model, `/config llm.context_size 8192`, or `/config llm.gpu_layers 20` |
| model ignores tools or loops | model too small; switch with `/model <id>` |
| search finds nothing semantic | `/download nomic-embed-v1.5` then `/index`; after changing the embedding model `/index --full` |

Logs: `data\logs\` (llama-server output, `mcp-operations.jsonl` with every MCP call).
