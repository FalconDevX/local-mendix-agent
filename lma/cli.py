"""Entry point.

  python main.py                      interactive terminal
  python main.py -c "/doctor"         run slash command(s) and exit (repeatable)
  python main.py -p "question" -m ask run one prompt and exit
  python main.py --project <dir> ...  open a project first
  --yes                               auto-approve permissions and apply staged edits (non-interactive runs)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from lma.config import load_settings, prepare_network_env
from lma.core.agent import Agent
from lma.core.state import AppState
from lma.ui.commands import MODES, Commands
from lma.ui.rendering import Renderer


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="lma", description="Local Mendix Agent")
    p.add_argument("-c", "--command", action="append", default=[], help="slash command to run, e.g. \"/mendix status\"")
    p.add_argument("-p", "--prompt", help="prompt to send to the agent")
    p.add_argument("-m", "--mode", choices=MODES, help="agent mode for -p")
    p.add_argument("--project", help="project folder to open")
    p.add_argument("--yes", action="store_true", help="approve tool permissions and apply staged edits automatically")
    p.add_argument("--no-mcp", action="store_true", help="do not connect MCP servers at startup (one-shot runs)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


async def one_shot(args: argparse.Namespace) -> int:
    settings = load_settings()
    state = AppState(settings)
    ui = Renderer(stream_markdown=False)
    state.emit = ui.emit
    state.permissions.auto_approve = args.yes
    agent = Agent(state)
    commands = Commands(state, agent, ui, ask=None, assume_yes=args.yes)
    code = 0
    try:
        if args.project:
            state.open_project(Path(args.project), persist=False)
        if not args.no_mcp:
            if state.mcp.connections:
                await state.mcp.connect_all()
            await state.mendix.connect()
            agent.toolbox.refresh_mcp()
        for line in args.command:
            await commands.dispatch(line)
        if args.prompt:
            res = await commands.chat(args.prompt, args.mode)
            code = 0 if res.stopped == "completed" else 1
    finally:
        ui.finish()
        await state.close()
    return code


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING, format="%(name)s: %(message)s")
    settings = load_settings()
    prepare_network_env(settings)
    try:
        if args.command or args.prompt:
            sys.exit(asyncio.run(one_shot(args)))
        from lma.ui.terminal import repl

        state = AppState(settings)
        if args.project:
            state.open_project(Path(args.project))
        asyncio.run(repl(state))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
