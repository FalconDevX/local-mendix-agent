# Test report - build machine, 2026-11-12

Machine: Windows 11, Python 3.12.6, i7-14700HX, 32 GB RAM, RTX 4060 Laptop 8 GB, ~6 GB free disk.
Runtime: llama.cpp b12307 `vulkan` build (the small one, because of the free disk; `cuda-12.4` is what
`/models` recommends for this GPU). No Mendix Studio Pro on this machine.

Models used: `qwen2.5-coder-1.5b` (chat), `qwen3-vl-4b` (vision, also used as chat), `nomic-embed-v1.5`.

## Automated suite

`python -m pytest -m "not maia"` -> **31 passed**, 3 deselected (the 3 live Maia tests).

## Acceptance tests

| # | Test | Result | How it was checked |
|---|---|---|---|
| 1 | Local inference, no Ollama/admin | PASS | runtime ZIP unpacked to `data\runtime`; streamed completion ~40-118 tok/s (1.5B), ~43-58 tok/s (4B); `test_local_model_inference_and_embeddings` |
| 2 | MCP connect/discover/execute | PASS | real MCP server process over stdio and Streamable HTTP; tools, resources, prompts, errors, reconnect after server restart |
| 3 | Maia connect + read | **NOT RUN** | no Studio Pro here. Diagnostics verified in the negative case (precise "not running / nothing listening" report). Client path verified against a non-Mendix MCP server only |
| 4 | Maia modification + verify | **NOT RUN** | same. A write-then-read-back was verified on the demo MCP server, which says nothing about Maia's tools |
| 5 | Index repo, multi-file questions | PASS | sample repo: "Which files import shop.auth.tokens and what do they use?" -> model called `file_dependencies`, answered `pricing.py` uses `current_user` (correct) |
| 6 | Multi-file edit, diff, apply, verify | PASS | real model changed `pricing.py` and `tests/test_pricing.py` via `edit_file`, unified diff shown, applied, `git diff --stat` = 2 files; plus unit test incl. undo and conflict detection |
| 7 | Memory after restart | PASS | new process, "What did we change in earlier sessions?" -> listed the discount change and the summary file from stored task records; unit test for project isolation |
| 8 | Vision on a Studio Pro screenshot | PASS on a **synthetic** image | a drawn domain-model picture (3 entities, 9 attributes, 2 associations, error CE1613): everything read correctly, both routed (text chat model + vision model) and direct. No real Studio Pro capture was available |
| 9 | Autonomous filesystem + MCP workflow | PASS | real model (4B): read README -> staged `docs/summary.md` -> `create_note` on the MCP server -> `list_notes` confirmed; file verified on disk. Also scripted-model test asserting ledger vs. real state |
| 10 | Offline | PASS (simulated) | external network blocked for the process (dead proxy): inference, search, memory, local MCP worked; Hugging Face/GitHub reported unreachable; `/download` failed with an explanation and no fallback. A physical network disconnect was not done |

## Findings worth knowing

- **Model size matters.** `qwen2.5-coder-1.5b` could answer questions but failed at agent work: it
  echoed tool definitions, invented an MCP server name and looped. `qwen3-vl-4b` (4B) handled every
  scenario above with native tool calls. For `/agent` and `/mendix` plan on `qwen3-8b` or
  `qwen2.5-coder-7b` at least; untested here because of disk space.
- Plans: the 4B model produced sensible plans but did not call `update_plan`, so step status stays
  "pending". The operations report is the reliable record.
- Download speed from Hugging Face varied between 0.02 and 20 MB/s during the build; resume worked
  across several interruptions and the final files passed SHA-256 verification.
- Do not run two downloads of the same model at once (two writers on one `.part` file corrupt it;
  the checksum catches it and the file is re-downloaded).

## Bugs found and fixed during testing

- `releases/latest` of llama.cpp is not always a binary build -> now picks the newest release that has the asset.
- `nvidia-smi` header format changed ("CUDA UMD Version") -> CUDA detection fixed.
- Stale HTTP MCP session after a server restart -> reconnect on failure (reads are retried, writes never).
- Tool calls written as text in several dialects -> tolerant parser, restricted to known tool names; echoed schemas rejected.
- `/models add` stored a relative path -> absolute now.
- `/vision` relayed the description through the text model with tools -> vision model answers directly.
- Keyword search could not find words inside `snake_case` identifiers -> tokenizer changed.
- llama-server could outlive a crashed parent -> Windows Job Object (verified with a hard exit).
- prompt_toolkit crashes without a real console (Git Bash/mintty, pipes) -> plain line-input fallback.

## Not covered

- Anything involving a real Studio Pro (tests 3, 4, real screenshots, Maia latency/timeouts, page tools).
- The `cuda-12.4` runtime variant (only `vulkan` was installed), models of 7B and larger.
- Corporate proxy with authentication, TLS inspection, AppLocker/WDAC - the code paths and messages
  exist, the environments do not exist here.
- The full-screen interactive prompt (completion, status bar, Ctrl+C cancel) was not driven by an
  automated test; the same command layer was exercised through `-c`/`-p` and the plain-input REPL.
