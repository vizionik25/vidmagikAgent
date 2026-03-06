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
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport

# Suppress litellm's noisy internal logs (set to True when debugging LLM issues)
litellm.suppress_debug_info = True

VIDMAGIK_DIR = Path(__file__).parent.parent.parent
MEDIA_DIR = VIDMAGIK_DIR / "media"

SYSTEM_PROMPT = """\
You are an expert AI video editor. The user will give you a video URL that has
already been downloaded to a local file. Your job is to create a compelling
short-form highlight reel (for TikTok, YouTube Shorts, Instagram Reels) from
the source video.

Workflow:
1. Load the video with `video_file_clip`.
2. Run `tools_detect_highlights` to find high-motion/action moments via optical flow.
   This returns a list of `{timestamp, intensity}` dicts.
3. **Pick the 3-5 most intense highlights.** For each one, create a subclip
   centred on that timestamp that is 5-15 seconds long.
   For example, if a highlight is at timestamp 45.2s, subclip from 40.0 to 50.0.
4. For each chosen segment (MAX 5):
   a. Extract it with `subclip` using start/end times that surround the highlight.
   b. Apply `vfx_auto_framing` with target_aspect_ratio=0.5625 for 9:16 vertical.
   c. Optionally add effects (fade_in, fade_out, etc.) as you see fit.
5. **Concatenate all the clips** into one video using `concatenate_video_clips`
   with the list of clip IDs from step 4.
6. Export the single concatenated clip with `write_videofile` to `media/short.mp4`.
7. Summarise what you created and why you chose those moments.

CRITICAL RULES:
- Select at MOST 5 segments, each 5-15 seconds long.
- ALWAYS concatenate your clips into ONE final video before exporting.
  Do NOT export each clip separately.
- All file paths must be relative to the project root (e.g. `media/video.mp4`).
- Output files go in the `media/` directory.
- Explain your creative decisions briefly.
- Do NOT ask the user questions — just proceed with your best judgement.
"""


def _get_llm_config() -> dict:
    """
    Resolve LLM configuration from environment variables.

    LM Studio (preferred for local LLMs):
        LM_STUDIO_API_BASE – e.g. "http://localhost:1234/v1"
        LM_STUDIO_API_KEY  – optional, default is empty
        LLM_MODEL          – model name WITHOUT prefix, e.g. "ibm/granite-4-h-tiny"
                             (will be auto-prefixed to "lm_studio/ibm/granite-4-h-tiny")

    Cloud providers (auto-detected from API key env vars):
        GEMINI_API_KEY     → gemini/gemini-2.0-flash
        OPENAI_API_KEY     → gpt-4o
        ANTHROPIC_API_KEY  → anthropic/claude-sonnet-4-20250514

    Explicit override:
        LLM_MODEL          – full litellm model string with provider prefix
        LLM_API_KEY        – API key
    """
    model = os.environ.get("LLM_MODEL", "")
    api_key = os.environ.get("LLM_API_KEY", "")
    lm_studio_base = os.environ.get("LM_STUDIO_API_BASE", "")

    # LM Studio: if LM_STUDIO_API_BASE is set, use the lm_studio/ provider
    if lm_studio_base:
        if model and not model.startswith("lm_studio/"):
            model = f"lm_studio/{model}"
        elif not model:
            model = "lm_studio/local-model"
        api_key = api_key or os.environ.get("LM_STUDIO_API_KEY", "")
        return {"model": model, "api_key": api_key}

    # Auto-detect from well-known provider env vars if no explicit config
    if not model:
        if os.environ.get("GEMINI_API_KEY"):
            model = "gemini/gemini-2.0-flash"
            api_key = api_key or os.environ["GEMINI_API_KEY"]
        elif os.environ.get("OPENAI_API_KEY"):
            model = "gpt-4o"
            api_key = api_key or os.environ["OPENAI_API_KEY"]
        elif os.environ.get("ANTHROPIC_API_KEY"):
            model = "anthropic/claude-sonnet-4-20250514"
            api_key = api_key or os.environ["ANTHROPIC_API_KEY"]

    return {"model": model, "api_key": api_key}


class MCPVideoClient:
    """Manages the vidmagik-mcp connection and the LLM agentic loop."""

    def __init__(self):
        # When MCP_SERVER_URL is set (e.g. in Docker Compose), connect over HTTP.
        # Otherwise fall back to spawning the server as a local subprocess.
        mcp_url = os.environ.get("MCP_SERVER_URL")
        if mcp_url:
            self._transport = StreamableHttpTransport(mcp_url)
        else:
            self._transport = StdioTransport(
                command="uv",
                args=["run", "src/api/main.py", "--transport", "stdio"],
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
        # Resolve final LLM config: UI values override env vars
        env_cfg = _get_llm_config()
        _model = model or env_cfg["model"]
        _api_key = api_key or env_cfg.get("api_key") or None

        # If the user passed an api_base via the UI, set it as LM_STUDIO_API_BASE
        # so litellm's lm_studio provider picks it up
        if api_base:
            os.environ["LM_STUDIO_API_BASE"] = api_base
            # Ensure model has lm_studio/ prefix when api_base is set via UI
            if _model and not _model.startswith(("lm_studio/", "gemini/", "openai/", "anthropic/", "ollama/", "azure/")):
                _model = f"lm_studio/{_model}"

        if not _model:
            yield {
                "type": "error",
                "text": (
                    "No LLM model configured. Set LM_STUDIO_API_BASE + LLM_MODEL env vars, "
                    "or set GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY, "
                    "or enter values in the UI settings."
                ),
            }
            return

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

        # Build kwargs for litellm – only include non-None values
        # NOTE: lm_studio provider reads LM_STUDIO_API_BASE from env,
        # so we do NOT pass api_base as a kwarg.
        llm_kwargs: dict = {
            "model": _model,
            "temperature": 0.3,
        }
        if _api_key:
            llm_kwargs["api_key"] = _api_key
        if self._openai_tools:
            llm_kwargs["tools"] = self._openai_tools
            llm_kwargs["tool_choice"] = "auto"

        yield {
            "type": "thinking",
            "text": f"Using model: {_model}",
        }

        # Loop until the LLM returns a final text response (no more tool calls)
        for _iteration in range(50):  # safety cap
            try:
                response = await litellm.acompletion(
                    messages=messages,
                    **llm_kwargs,
                )
            except litellm.APIConnectionError as exc:
                endpoint = _api_base or "default provider endpoint"
                yield {
                    "type": "error",
                    "text": (
                        f"Connection error: Cannot reach {endpoint}. "
                        f"Make sure the LLM server is running and accessible. "
                        f"Details: {exc}"
                    ),
                }
                return
            except litellm.AuthenticationError as exc:
                yield {
                    "type": "error",
                    "text": (
                        f"Authentication error: Invalid API key for model '{_model}'. "
                        f"Check LLM_API_KEY env var or the API Key field in settings. "
                        f"Details: {exc}"
                    ),
                }
                return
            except litellm.NotFoundError as exc:
                yield {
                    "type": "error",
                    "text": (
                        f"Model not found: '{_model}' is not available. "
                        f"Check the model name. Details: {exc}"
                    ),
                }
                return
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
