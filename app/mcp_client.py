"""
MCP client + LLM agentic loop for vidmagik-mcp.

Connects to vidmagik-mcp over stdio, converts MCP tool schemas to OpenAI
function-calling format, and runs an agentic loop where LiteLLM (any
OpenAI-compatible provider) orchestrates all tool calls.
"""

import json
import os
from pathlib import Path
from typing import AsyncIterator

import litellm
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

VIDMAGIK_DIR = Path(__file__).parent.parent
MEDIA_DIR = VIDMAGIK_DIR / "media"

SYSTEM_PROMPT = """\
You are an expert AI video editor. The user will give you a video URL that has
already been downloaded to a local file. Your job is to create compelling
short-form video clips (for TikTok, YouTube Shorts, Instagram Reels) from the
source video.

Workflow:
1. Load the video with `video_file_clip`.
2. Run `tools_detect_scenes` to find scene boundaries.
3. Analyse the scene list and pick the 1-3 most promising segments for shorts
   (aim for 15-60 seconds each).
4. For each chosen segment:
   a. Extract it with `subclip`.
   b. Apply `vfx_auto_framing` with target_aspect_ratio=0.5625 for 9:16 vertical.
   c. Add tasteful effects (fade_in, fade_out, etc.) as you see fit.
   d. Export with `write_videofile` to `media/short_<N>.mp4`.
5. When finished, summarise what you created and why you chose those moments.

Important rules:
- All file paths must be relative to the project root (e.g. `media/video.mp4`).
- Output files go in the `media/` directory.
- Explain your creative decisions briefly.
- Do NOT ask the user questions — just proceed with your best judgement.
"""


class MCPVideoClient:
    """Manages the vidmagik-mcp subprocess and the LLM agentic loop."""

    def __init__(self):
        self._transport = StdioTransport(
            command="uv",
            args=["run", "api/main.py", "--transport", "stdio"],
            cwd=str(VIDMAGIK_DIR),
        )
        self._client = Client(self._transport)
        self._connected = False
        self._openai_tools: list[dict] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self):
        await self._client.__aenter__()
        self._connected = True
        await self._load_tool_schemas()

    async def disconnect(self):
        if self._connected:
            await self._client.__aexit__(None, None, None)
            self._connected = False

    # ------------------------------------------------------------------
    # MCP tool schemas → OpenAI function calling format
    # ------------------------------------------------------------------

    async def _load_tool_schemas(self):
        """Fetch all MCP tools and convert to OpenAI function-calling format."""
        tools = await self._client.list_tools()
        self._openai_tools = []
        for tool in tools:
            fn: dict = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}},
                },
            }
            self._openai_tools.append(fn)

    # ------------------------------------------------------------------
    # Execute a single MCP tool call
    # ------------------------------------------------------------------

    async def _call_tool(self, name: str, arguments: dict) -> str:
        result = await self._client.call_tool(name, arguments)
        for block in result.content:
            if hasattr(block, "text"):
                return block.text
        return str(result.data) if result.data is not None else ""

    # ------------------------------------------------------------------
    # Video download (yt-dlp, runs in-process)
    # ------------------------------------------------------------------

    def download_video(self, url: str) -> str:
        import yt_dlp

        MEDIA_DIR.mkdir(exist_ok=True)
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(MEDIA_DIR / "%(title)s.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    async def run_agent(
        self,
        video_path: str,
        user_message: str,
        *,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> AsyncIterator[dict]:
        """
        Run the agentic loop. Yields dicts for the UI to render:

            {"type": "thinking",   "text": "..."}
            {"type": "tool_call",  "name": "...", "args": {...}}
            {"type": "tool_result","name": "...", "result": "..."}
            {"type": "message",    "text": "..."}
            {"type": "error",      "text": "..."}
        """
        rel_path = os.path.relpath(video_path, str(VIDMAGIK_DIR))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"The video has been downloaded to `{rel_path}`. "
                    f"{user_message}"
                ),
            },
        ]

        # Loop until the LLM returns a final text response (no more tool calls)
        for _iteration in range(50):  # safety cap
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    tools=self._openai_tools if self._openai_tools else None,
                    tool_choice="auto",
                    api_base=api_base,
                    api_key=api_key,
                    temperature=0.3,
                )
            except Exception as exc:
                yield {"type": "error", "text": f"LLM error: {exc}"}
                return

            choice = response.choices[0]
            msg = choice.message

            # If the LLM produced text content, stream it
            if msg.content:
                yield {"type": "thinking", "text": msg.content}

            # If no tool calls, we're done
            if not msg.tool_calls:
                if msg.content:
                    yield {"type": "message", "text": msg.content}
                return

            # Append the assistant message (with tool_calls) to history
            messages.append(msg.model_dump())

            # Execute each tool call
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                yield {"type": "tool_call", "name": fn_name, "args": fn_args}

                try:
                    result = await self._call_tool(fn_name, fn_args)
                except Exception as exc:
                    result = f"Error: {exc}"

                yield {"type": "tool_result", "name": fn_name, "result": result}

                # Append tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        yield {"type": "error", "text": "Agent reached maximum iterations."}
