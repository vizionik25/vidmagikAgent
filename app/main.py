"""AI Shorts Creator – NiceGUI app with LLM-orchestrated MCP tool calling."""

import asyncio
import os
from pathlib import Path

from nicegui import ui, app, run

from mcp_client import MCPVideoClient, MEDIA_DIR

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

mcp = MCPVideoClient()

settings: dict = {
    "api_base": "http://localhost:1234/v1",
    "api_key": "lm-studio",
    "model": "local-model",
}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def _startup():
    MEDIA_DIR.mkdir(exist_ok=True)
    await mcp.connect()


async def _shutdown():
    await mcp.disconnect()


app.on_startup(_startup)
app.on_shutdown(_shutdown)

MEDIA_DIR.mkdir(exist_ok=True)
app.add_static_files("/media", str(MEDIA_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _icon_for(event_type: str) -> str:
    return {
        "thinking": "psychology",
        "tool_call": "build",
        "tool_result": "check_circle",
        "message": "smart_toy",
        "error": "error",
    }.get(event_type, "info")


def _color_for(event_type: str) -> str:
    return {
        "thinking": "text-gray-400",
        "tool_call": "text-violet-400",
        "tool_result": "text-cyan-400",
        "message": "text-green-400",
        "error": "text-red-400",
    }.get(event_type, "text-gray-400")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


@ui.page("/")
async def index():
    ui.dark_mode(True)
    ui.colors(
        primary="#7c3aed",
        secondary="#06b6d4",
        accent="#a78bfa",
        positive="#10b981",
        negative="#ef4444",
    )

    # -- Header --
    with ui.header().classes("items-center justify-between px-6"):
        ui.icon("movie_filter", size="sm").classes("text-violet-400")
        ui.label("AI Shorts Creator").classes("text-xl font-bold tracking-wide")
        ui.space()
        ui.label("powered by vidmagik-mcp").classes("text-xs text-gray-400")

    # -- Main layout --
    with ui.column().classes("w-full max-w-5xl mx-auto px-4 py-6 gap-6"):

        # ============================================================
        # Settings panel (collapsible)
        # ============================================================
        with ui.expansion("LLM Settings", icon="settings").classes(
            "w-full bg-gray-900 rounded-xl"
        ).props("default-opened"):
            with ui.row().classes("w-full gap-4 flex-wrap"):
                ui.input(
                    label="API Base URL",
                    placeholder="http://localhost:1234/v1",
                ).bind_value(settings, "api_base").props(
                    "outlined dark rounded"
                ).classes("flex-1 min-w-[250px]")

                ui.input(
                    label="API Key",
                    placeholder="sk-... or lm-studio",
                ).bind_value(settings, "api_key").props(
                    "outlined dark rounded"
                ).classes("flex-1 min-w-[200px]")

                ui.input(
                    label="Model",
                    placeholder="gpt-4o / local-model / etc.",
                ).bind_value(settings, "model").props(
                    "outlined dark rounded"
                ).classes("flex-1 min-w-[200px]")

        # ============================================================
        # Video URL input
        # ============================================================
        with ui.card().classes("w-full bg-gray-900 rounded-xl"):
            ui.label("Video Source").classes("text-lg font-semibold mb-2")
            with ui.row().classes("w-full gap-2 items-end"):
                url_input = ui.input(
                    label="Video URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                ).props("outlined dark rounded").classes("flex-grow")

                instructions_input = ui.textarea(
                    label="Instructions (optional)",
                    placeholder="e.g. Focus on the funniest moments, make 3 shorts...",
                ).props("outlined dark rounded rows=2").classes("flex-grow")

            with ui.row().classes("mt-4 gap-2"):
                go_button = ui.button(
                    "Create Shorts", icon="auto_awesome",
                ).props("rounded unelevated size=lg")

        # ============================================================
        # Agent activity log (chat-style)
        # ============================================================
        with ui.card().classes("w-full bg-gray-900 rounded-xl"):
            ui.label("Agent Activity").classes("text-lg font-semibold mb-2")
            log_container = ui.column().classes("w-full gap-1 max-h-[500px] overflow-y-auto")
            status_label = ui.label("Waiting for input…").classes(
                "text-sm text-gray-500 mt-2"
            )

        # ============================================================
        # Downloads area
        # ============================================================
        download_card = ui.card().classes("w-full bg-gray-900 rounded-xl")
        download_card.visible = False
        with download_card:
            ui.label("Exported Shorts").classes("text-lg font-semibold mb-2")
            download_container = ui.column().classes("w-full gap-2")

        # ============================================================
        # Logic
        # ============================================================

        async def on_go():
            url = url_input.value.strip()
            if not url:
                ui.notify("Enter a video URL", type="warning")
                return
            if not settings["model"].strip():
                ui.notify("Set a model name in LLM Settings", type="warning")
                return

            go_button.disable()
            log_container.clear()
            download_container.clear()
            download_card.visible = False

            # -- Step 1: Download --
            status_label.text = "⬇️  Downloading video…"
            with log_container:
                with ui.row().classes("items-start gap-2"):
                    ui.icon("download").classes("text-violet-400 mt-1")
                    ui.label(f"Downloading: {url}").classes("text-gray-300")

            try:
                video_path = await run.io_bound(mcp.download_video, url)
            except Exception as exc:
                status_label.text = f"❌ Download failed: {exc}"
                ui.notify(str(exc), type="negative")
                go_button.enable()
                return

            filename = Path(video_path).name
            with log_container:
                with ui.row().classes("items-start gap-2"):
                    ui.icon("check_circle").classes("text-green-400 mt-1")
                    ui.label(f"Downloaded → {filename}").classes("text-green-300")

            # -- Step 2: Run agent --
            status_label.text = "🤖  Agent is working…"
            user_msg = instructions_input.value.strip() or "Create the best shorts from this video."

            async for event in mcp.run_agent(
                video_path=video_path,
                user_message=user_msg,
                model=settings["model"].strip(),
                api_base=settings["api_base"].strip() or None,
                api_key=settings["api_key"].strip() or None,
            ):
                etype = event["type"]
                icon = _icon_for(etype)
                color = _color_for(etype)

                with log_container:
                    if etype == "tool_call":
                        with ui.row().classes("items-start gap-2"):
                            ui.icon(icon).classes(f"{color} mt-1")
                            with ui.column().classes("gap-0"):
                                ui.label(f"Calling: {event['name']}").classes(
                                    f"{color} font-semibold text-sm"
                                )
                                args_str = ", ".join(
                                    f"{k}={v!r}" for k, v in event["args"].items()
                                )
                                if args_str:
                                    ui.label(args_str).classes(
                                        "text-xs text-gray-500 font-mono"
                                    )

                    elif etype == "tool_result":
                        with ui.row().classes("items-start gap-2"):
                            ui.icon(icon).classes(f"{color} mt-1")
                            result_text = str(event["result"])[:200]
                            ui.label(f"→ {result_text}").classes(
                                "text-xs text-gray-400 font-mono"
                            )
                        # Check if export produced a file
                        if "short" in event.get("name", "") or "write" in event.get("name", ""):
                            _check_for_downloads(download_container, download_card, event["result"])

                    elif etype == "message":
                        with ui.card().classes("w-full bg-gray-800 mt-2"):
                            ui.markdown(event["text"]).classes("text-gray-200")

                    elif etype == "error":
                        with ui.row().classes("items-start gap-2"):
                            ui.icon(icon).classes(f"{color} mt-1")
                            ui.label(event["text"]).classes(f"{color} text-sm")

                    elif etype == "thinking":
                        with ui.row().classes("items-start gap-2"):
                            ui.icon(icon).classes(f"{color} mt-1")
                            ui.label(event["text"][:300]).classes(
                                "text-xs text-gray-500"
                            )

                # Scroll log to bottom
                await ui.run_javascript(
                    "document.querySelector('.max-h-\\\\[500px\\\\]')?.scrollTo(0, 999999)"
                )

            # Scan media dir for any generated shorts
            _scan_for_shorts(download_container, download_card)

            status_label.text = "✅  Agent finished!"
            ui.notify("All done!", type="positive")
            go_button.enable()

        go_button.on_click(on_go)


def _check_for_downloads(container, card, result_text: str):
    """Check if a tool result mentions a written file and add download link."""
    if "Successfully wrote video" in str(result_text):
        card.visible = True
        # Extract filename from result
        parts = str(result_text).split("to ")
        if len(parts) > 1:
            fname = parts[-1].strip()
            fpath = MEDIA_DIR.parent / fname if not fname.startswith("/") else Path(fname)
            if fpath.exists():
                with container:
                    with ui.card().classes("w-full bg-gray-800"):
                        with ui.row().classes("items-center"):
                            ui.icon("movie").classes("text-green-400 text-xl")
                            ui.label(fpath.name).classes("font-semibold")
                            ui.space()
                            ui.button(
                                "Download", icon="download",
                                on_click=lambda p=str(fpath): ui.download(p),
                            ).props("flat rounded color=positive size=sm")


def _scan_for_shorts(container, card):
    """Scan media/ for any short_*.mp4 files and add download links."""
    for f in sorted(MEDIA_DIR.glob("short*.mp4")):
        card.visible = True
        with container:
            with ui.card().classes("w-full bg-gray-800"):
                with ui.row().classes("items-center"):
                    ui.icon("movie").classes("text-green-400 text-xl")
                    ui.label(f.name).classes("font-semibold")
                    ui.space()
                    ui.button(
                        "Download", icon="download",
                        on_click=lambda p=str(f): ui.download(p),
                    ).props("flat rounded color=positive size=sm")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ui.run(
        title="AI Shorts Creator",
        host=os.environ.get("NICEGUI_HOST", "127.0.0.1"),
        port=3000,
        dark=True,
        reload=False,
        favicon="🎬",
    )
