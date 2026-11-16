"""Sub-agent roles: what each kind of sub-agent may do and how it is instructed."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    summary: str      # one line, shown to the lead model and in /agents
    mode: str         # permission mode the sub-agent runs in (decides which tools it gets)
    prompt: str
    phase: int        # /team order: 1 = investigate, 2 = build, 3 = check


REPORT_RULES = (
    "You are a sub-agent. Nobody sees your intermediate steps: your LAST message is the only thing handed back, "
    "so make it a complete, factual report. Name files as path:line. Say plainly what you could not find or do. "
    "Do not ask questions; decide with what you have. Stop as soon as the task is covered.")

ROLES: dict[str, Role] = {
    "explorer": Role(
        "read-only investigation of the code (where is X, how does Y work)", "ask",
        "ROLE explorer: investigate the project with the read and search tools and answer the task. Never guess: "
        "open the files. Report the relevant files, functions and how they connect.", 1),
    "coder": Role(
        "implements a change with staged file edits", "code",
        "ROLE coder: implement exactly the described change. Read each file before editing it, keep the existing "
        "style, make minimal edits with edit_file/write_file. Report which files you staged and what changed.", 2),
    "mendix": Role(
        "works on the Mendix app in Studio Pro through the Maia MCP tools", "mendix",
        "ROLE mendix: carry out the task in the Mendix app using only the MCP tools the server really offers. "
        "Inspect before modifying. Report every operation and its real result.", 2),
    "reviewer": Role(
        "read-only review of staged changes or existing code", "review",
        "ROLE reviewer: review what the task names (staged edits are visible through read_file). Report concrete "
        "problems as `file:line - severity - issue - suggested fix`, most severe first, or state that you found none.", 3),
}
