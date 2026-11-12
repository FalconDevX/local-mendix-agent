"""Permission model for every tool call the agent makes.

Risk classes:
  read    - no side effects: always allowed
  stage   - only stages a file change for later review: allowed (nothing is written yet)
  write   - changes something outside the staging area (MCP modification, memory): needs approval
  exec    - runs a process or deletes/commits: needs approval every time unless allow-listed
Mode restrictions come on top: /ask, /plan and /review can never use write or exec tools.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

READ_ONLY_MODES = {"ask", "plan", "review", "vision"}

# (tool name, risk, human readable summary) -> "y" | "n" | "a" (always for this session)
Approver = Callable[[str, str, str], Awaitable[str]]


@dataclass
class Decision:
    allowed: bool
    reason: str = ""


@dataclass
class Permissions:
    mode: str = "agent"
    approver: Approver | None = None
    auto_approve: bool = False  # non-interactive runs started with --yes
    session_allow: set[str] = field(default_factory=set)
    denied_log: list[str] = field(default_factory=list)

    async def check(self, tool: str, risk: str, summary: str) -> Decision:
        if risk in ("read", "stage"):
            if risk == "stage" and self.mode in READ_ONLY_MODES:
                return self._deny(tool, f"/{self.mode} mode is read-only; switch to /code or /agent to edit files")
            return Decision(True)
        if self.mode in READ_ONLY_MODES:
            return self._deny(tool, f"/{self.mode} mode is read-only; this tool modifies state")
        if tool in self.session_allow or self.auto_approve:
            return Decision(True)
        if self.approver is None:
            return self._deny(tool, "no approval channel available (non-interactive run without --yes)")
        answer = await self.approver(tool, risk, summary)
        if answer == "a":
            self.session_allow.add(tool)
            return Decision(True)
        if answer == "y":
            return Decision(True)
        return self._deny(tool, "the user declined this operation")

    def _deny(self, tool: str, reason: str) -> Decision:
        self.denied_log.append(f"{tool}: {reason}")
        return Decision(False, reason)
