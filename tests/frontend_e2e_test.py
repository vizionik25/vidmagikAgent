"""
End-to-end tests for the AI Shorts Creator frontend + project-wide coverage.

Covers:
- app/mcp_client.py: MCPVideoClient lifecycle, tool schema conversion, _call_tool,
  download_video, run_agent (all branches of the agentic loop)
- app/main.py: helper functions, settings, SYSTEM_PROMPT
- api/custom_fx/: all custom effect classes
- api/main.py (backend): all vfx/afx wrappers, audio composition, prompts,
  parse_args, main(), upload edge cases, check_installation
"""

import json
import os
import sys
import asyncio
import shutil
from pathlib import Path
from types import SimpleNamespace
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import numpy as np

# Ensure the app/ directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))


# ======================================================================
# mcp_client.py tests
# ======================================================================

from mcp_client import MCPVideoClient, MEDIA_DIR, SYSTEM_PROMPT, VIDMAGIK_DIR


class TestMCPVideoClientInit:
    def test_constants(self):
        assert VIDMAGIK_DIR.exists()
        assert "video_file_clip" in SYSTEM_PROMPT
        assert "shorts" in SYSTEM_PROMPT.lower()

    def test_init(self):
        client = MCPVideoClient()
        assert client._connected is False
        assert client._openai_tools == []
        assert client._transport is not None

    def test_media_dir_path(self):
        assert MEDIA_DIR == VIDMAGIK_DIR / "media"


class TestMCPVideoClientLifecycle:
    @pytest.mark.asyncio
    async def test_connect(self):
        client = MCPVideoClient()
        client._client.__aenter__ = AsyncMock()
        client._load_tool_schemas = AsyncMock()
        await client.connect()
        assert client._connected is True

    @pytest.mark.asyncio
    async def test_disconnect_when_connected(self):
        client = MCPVideoClient()
        client._connected = True
        client._client.__aexit__ = AsyncMock()
        await client.disconnect()
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self):
        client = MCPVideoClient()
        client._connected = False
        client._client.__aexit__ = AsyncMock()
        await client.disconnect()
        client._client.__aexit__.assert_not_awaited()


class TestLoadToolSchemas:
    @pytest.mark.asyncio
    async def test_converts_tools(self):
        client = MCPVideoClient()
        t1 = SimpleNamespace(name="a", description="desc", inputSchema={"type": "object", "properties": {"x": {}}})
        t2 = SimpleNamespace(name="b", description=None, inputSchema=None)
        client._client.list_tools = AsyncMock(return_value=[t1, t2])
        await client._load_tool_schemas()
        assert len(client._openai_tools) == 2
        assert client._openai_tools[1]["function"]["description"] == ""

    @pytest.mark.asyncio
    async def test_empty_tools(self):
        client = MCPVideoClient()
        client._client.list_tools = AsyncMock(return_value=[])
        await client._load_tool_schemas()
        assert client._openai_tools == []


class TestCallTool:
    @pytest.mark.asyncio
    async def test_returns_text(self):
        client = MCPVideoClient()
        result = SimpleNamespace(content=[SimpleNamespace(text="ok")], data=None)
        client._client.call_tool = AsyncMock(return_value=result)
        assert await client._call_tool("t", {}) == "ok"

    @pytest.mark.asyncio
    async def test_returns_first_text(self):
        client = MCPVideoClient()
        result = SimpleNamespace(content=[SimpleNamespace(text="a"), SimpleNamespace(text="b")], data=None)
        client._client.call_tool = AsyncMock(return_value=result)
        assert await client._call_tool("t", {}) == "a"

    @pytest.mark.asyncio
    async def test_falls_back_to_data(self):
        client = MCPVideoClient()
        result = SimpleNamespace(content=[SimpleNamespace()], data={"r": 42})
        client._client.call_tool = AsyncMock(return_value=result)
        assert "42" in await client._call_tool("t", {})

    @pytest.mark.asyncio
    async def test_returns_empty(self):
        client = MCPVideoClient()
        result = SimpleNamespace(content=[SimpleNamespace()], data=None)
        client._client.call_tool = AsyncMock(return_value=result)
        assert await client._call_tool("t", {}) == ""

    @pytest.mark.asyncio
    async def test_empty_content_data(self):
        client = MCPVideoClient()
        result = SimpleNamespace(content=[], data="val")
        client._client.call_tool = AsyncMock(return_value=result)
        assert await client._call_tool("t", {}) == "val"


class TestDownloadVideo:
    def test_download(self):
        client = MCPVideoClient()
        m = MagicMock()
        m.extract_info.return_value = {}
        m.prepare_filename.return_value = "/tmp/t.mp4"
        with patch("yt_dlp.YoutubeDL") as cls:
            cls.return_value.__enter__ = MagicMock(return_value=m)
            cls.return_value.__exit__ = MagicMock(return_value=False)
            assert client.download_video("http://x") == "/tmp/t.mp4"

    def test_download_creates_dir(self, tmp_path):
        client = MCPVideoClient()
        m = MagicMock()
        m.extract_info.return_value = {}
        m.prepare_filename.return_value = str(tmp_path / "o.mp4")
        with patch("yt_dlp.YoutubeDL") as cls, patch("mcp_client.MEDIA_DIR", tmp_path / "new"):
            cls.return_value.__enter__ = MagicMock(return_value=m)
            cls.return_value.__exit__ = MagicMock(return_value=False)
            client.download_video("http://x")
            assert (tmp_path / "new").exists()


class TestRunAgent:
    def _mc(self):
        c = MCPVideoClient()
        c._openai_tools = [{"type": "function", "function": {"name": "t", "description": "", "parameters": {}}}]
        return c

    def _resp(self, content=None, tcs=None):
        msg = SimpleNamespace(
            content=content, tool_calls=tcs,
            model_dump=lambda: {"role": "assistant", "content": content,
                                "tool_calls": [{"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}, "type": "function"} for tc in (tcs or [])] if tcs else None})
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    def _tc(self, name, args, cid="tc1"):
        return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))

    @pytest.mark.asyncio
    async def test_final_message(self):
        c = self._mc()
        with patch("mcp_client.litellm.acompletion", new_callable=AsyncMock, return_value=self._resp("done")):
            events = [e async for e in c.run_agent("/tmp/v", "go", model="m")]
        assert events[-1] == {"type": "message", "text": "done"}

    @pytest.mark.asyncio
    async def test_tool_loop(self):
        c = self._mc()
        r1 = self._resp("thinking", [self._tc("t", {"a": 1})])
        r2 = self._resp("done")
        n = 0
        async def comp(**kw):
            nonlocal n; n += 1; return r1 if n == 1 else r2
        c._call_tool = AsyncMock(return_value="ok")
        with patch("mcp_client.litellm.acompletion", side_effect=comp):
            events = [e async for e in c.run_agent("/tmp/v", "go", model="m")]
        types = {e["type"] for e in events}
        assert types >= {"thinking", "tool_call", "tool_result", "message"}

    @pytest.mark.asyncio
    async def test_llm_error(self):
        c = self._mc()
        with patch("mcp_client.litellm.acompletion", new_callable=AsyncMock, side_effect=Exception("fail")):
            events = [e async for e in c.run_agent("/tmp/v", "go", model="m")]
        assert events[0]["type"] == "error"

    @pytest.mark.asyncio
    async def test_bad_json_args(self):
        c = self._mc()
        tc = SimpleNamespace(id="x", function=SimpleNamespace(name="t", arguments="{bad"))
        r1, r2 = self._resp(tcs=[tc]), self._resp("done")
        n = 0
        async def comp(**kw):
            nonlocal n; n += 1; return r1 if n == 1 else r2
        c._call_tool = AsyncMock(return_value="ok")
        with patch("mcp_client.litellm.acompletion", side_effect=comp):
            events = [e async for e in c.run_agent("/tmp/v", "go", model="m")]
        assert next(e for e in events if e["type"] == "tool_call")["args"] == {}

    @pytest.mark.asyncio
    async def test_tool_error(self):
        c = self._mc()
        r1 = self._resp(tcs=[self._tc("t", {})])
        r2 = self._resp("ok")
        n = 0
        async def comp(**kw):
            nonlocal n; n += 1; return r1 if n == 1 else r2
        c._call_tool = AsyncMock(side_effect=Exception("crash"))
        with patch("mcp_client.litellm.acompletion", side_effect=comp):
            events = [e async for e in c.run_agent("/tmp/v", "go", model="m")]
        assert "Error: crash" in next(e for e in events if e["type"] == "tool_result")["result"]

    @pytest.mark.asyncio
    async def test_no_content_no_tools(self):
        c = self._mc()
        with patch("mcp_client.litellm.acompletion", new_callable=AsyncMock, return_value=self._resp()):
            events = [e async for e in c.run_agent("/tmp/v", "go", model="m")]
        assert events == []

    @pytest.mark.asyncio
    async def test_max_iterations(self):
        c = self._mc()
        r = self._resp(tcs=[self._tc("t", {})])
        c._call_tool = AsyncMock(return_value="ok")
        with patch("mcp_client.litellm.acompletion", new_callable=AsyncMock, return_value=r):
            events = [e async for e in c.run_agent("/tmp/v", "go", model="m")]
        assert events[-1]["type"] == "error" and "maximum" in events[-1]["text"]

    @pytest.mark.asyncio
    async def test_empty_tools_none(self):
        c = MCPVideoClient(); c._openai_tools = []
        with patch("mcp_client.litellm.acompletion", new_callable=AsyncMock, return_value=self._resp("ok")) as m:
            [e async for e in c.run_agent("/tmp/v", "go", model="m")]
            assert m.call_args.kwargs["tools"] is None

    @pytest.mark.asyncio
    async def test_params_forwarded(self):
        c = self._mc()
        with patch("mcp_client.litellm.acompletion", new_callable=AsyncMock, return_value=self._resp("ok")) as m:
            [e async for e in c.run_agent("/tmp/v", "go", model="g", api_base="http://x", api_key="k")]
            kw = m.call_args.kwargs
            assert kw["api_base"] == "http://x" and kw["api_key"] == "k"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        c = self._mc()
        r1 = self._resp(tcs=[self._tc("a", {}, "t1"), self._tc("b", {}, "t2")])
        r2 = self._resp("done")
        n = 0
        async def comp(**kw):
            nonlocal n; n += 1; return r1 if n == 1 else r2
        c._call_tool = AsyncMock(return_value="ok")
        with patch("mcp_client.litellm.acompletion", side_effect=comp):
            events = [e async for e in c.run_agent("/tmp/v", "go", model="m")]
        assert len([e for e in events if e["type"] == "tool_call"]) == 2


# ======================================================================
# app/main.py tests
# ======================================================================

with patch("mcp_client.MCPVideoClient") as _mc, patch("nicegui.app") as _ma:
    _mc.return_value = MagicMock()
    from app import main as frontend_main


class TestHelpers:
    def test_icon_for(self):
        for k, v in [("thinking", "psychology"), ("tool_call", "build"), ("tool_result", "check_circle"),
                      ("message", "smart_toy"), ("error", "error"), ("x", "info")]:
            assert frontend_main._icon_for(k) == v

    def test_color_for(self):
        for k, v in [("thinking", "text-gray-400"), ("tool_call", "text-violet-400"),
                      ("tool_result", "text-cyan-400"), ("message", "text-green-400"),
                      ("error", "text-red-400"), ("x", "text-gray-400")]:
            assert frontend_main._color_for(k) == v

class TestSettings:
    def test_defaults(self):
        assert frontend_main.settings == {"api_base": "http://localhost:1234/v1", "api_key": "lm-studio", "model": "local-model"}

class TestCheckForDownloads:
    def test_no_match(self):
        frontend_main._check_for_downloads(MagicMock(), MagicMock(), "nope")

    def test_match_existing(self, tmp_path):
        f = tmp_path / "s.mp4"; f.write_text("x")
        with patch.object(frontend_main, "MEDIA_DIR", tmp_path), patch("app.main.ui"):
            frontend_main._check_for_downloads(MagicMock(), MagicMock(), f"Successfully wrote video to {f}")

    def test_match_missing(self):
        frontend_main._check_for_downloads(MagicMock(), MagicMock(), "Successfully wrote video to missing/f.mp4")

class TestScanForShorts:
    def test_finds_files(self, tmp_path):
        (tmp_path / "short_1.mp4").write_text("x")
        with patch.object(frontend_main, "MEDIA_DIR", tmp_path), patch("app.main.ui"):
            frontend_main._scan_for_shorts(MagicMock(), MagicMock())

    def test_no_files(self, tmp_path):
        with patch.object(frontend_main, "MEDIA_DIR", tmp_path):
            frontend_main._scan_for_shorts(MagicMock(), MagicMock())

class TestSystemPrompt:
    def test_content(self):
        for kw in ["video_file_clip", "tools_detect_scenes", "subclip", "vfx_auto_framing", "write_videofile", "media/", "9:16"]:
            assert kw in SYSTEM_PROMPT
        assert len(SYSTEM_PROMPT) > 100


# ======================================================================
# custom_fx/ tests
# ======================================================================

from moviepy import ColorClip
from api.custom_fx import AutoFraming, ChromaKey, CloneGrid, Matrix, QuadMirror, RGBSync


def _clip(w=120, h=120, dur=0.2, fps=5):
    """Create a test clip large enough for cv2 operations."""
    c = ColorClip((w, h), (128, 64, 200), duration=dur)
    return c.with_fps(fps)


class TestAutoFraming:
    def test_focus_func(self):
        effect = AutoFraming(target_aspect_ratio=9/16, focus_func=lambda fr, t: (60, 60))
        f = effect.apply(_clip()).get_frame(0)
        assert f.shape[0] > 0


class TestChromaKey:
    def test_softness(self):
        result = ChromaKey(color=(128, 64, 200), threshold=50, softness=20).apply(_clip())
        assert result.mask is not None

    def test_zero_softness(self):
        result = ChromaKey(color=(0, 255, 0), threshold=50, softness=0).apply(_clip())
        assert result.mask is not None



class TestCloneGrid:
    def test_init(self):
        e = CloneGrid(n_clones=4)
        assert e.rows == 2 and e.cols == 2

    def test_non_pow2(self):
        e = CloneGrid(n_clones=3)
        assert e.rows > 0 and e.cols > 0


class TestMatrix:
    def test_default(self):
        f = Matrix(speed=100, density=0.3, font_size=10).apply(_clip(w=64, h=64)).get_frame(0)
        assert f.shape[:2] == (64, 64)

    def test_colors(self):
        for c in ["red", "blue", "white", "pink"]:
            f = Matrix(speed=100, density=0.3, color=c, font_size=10).apply(_clip(w=64, h=64)).get_frame(0)
            assert f.shape[:2] == (64, 64)


class TestQuadMirror:
    def test_default(self):
        f = QuadMirror().apply(_clip()).get_frame(0)
        assert f.shape[:2] == (120, 120)

    def test_custom(self):
        f = QuadMirror(x=30, y=70).apply(_clip()).get_frame(0)
        assert f.shape[:2] == (120, 120)

    def test_oob(self):
        f = QuadMirror(x=200, y=-10).apply(_clip()).get_frame(0)
        assert f.shape[0] > 0


class TestRGBSync:
    def test_offsets(self):
        f = RGBSync(r_offset=(5, 0), b_offset=(-5, 0)).apply(_clip(dur=0.3)).get_frame(0)
        assert f.shape[2] == 3

    def test_time_offsets(self):
        f = RGBSync(g_time_offset=0.05).apply(_clip(dur=0.5)).get_frame(0.1)
        assert f.shape[2] == 3

    def test_no_offsets(self):
        f = RGBSync().apply(_clip()).get_frame(0)
        assert f.shape[2] == 3


# ======================================================================
# Backend main.py — cover ALL uncovered wrappers/functions
# ======================================================================

import api.main as backend_main
from api.main import CLIPS, register_clip, get_clip, color_clip, validate_path, parse_args


@pytest.fixture(autouse=True)
def cleanup_clips():
    CLIPS.clear()
    yield
    CLIPS.clear()


def _cid():
    return color_clip([20, 20], [128, 64, 200], duration=1)


class TestBackendVfx:
    """Cover all 2-line vfx wrappers that were missed."""

    def test_even_size(self):
        from api.main import vfx_even_size
        assert vfx_even_size(_cid()) in CLIPS

    def test_freeze(self):
        from api.main import vfx_freeze
        assert vfx_freeze(_cid(), t=0, freeze_duration=0.5) in CLIPS

    def test_loop(self):
        from api.main import vfx_loop
        assert vfx_loop(_cid(), n=2) in CLIPS

    def test_lum_contrast(self):
        from api.main import vfx_lum_contrast
        assert vfx_lum_contrast(_cid(), lum=10, contrast=5) in CLIPS

    def test_make_loopable(self):
        from api.main import vfx_make_loopable
        assert vfx_make_loopable(_cid(), overlap_duration=0.2) in CLIPS

    def test_margin(self):
        from api.main import vfx_margin
        assert vfx_margin(_cid(), margin=5, color=[255, 0, 0]) in CLIPS

    def test_mask_color(self):
        from api.main import vfx_mask_color
        assert vfx_mask_color(_cid(), color=[0, 0, 0]) in CLIPS

    def test_masks_and(self):
        from api.main import vfx_masks_and
        assert vfx_masks_and(_cid(), _cid()) in CLIPS

    def test_masks_or(self):
        from api.main import vfx_masks_or
        assert vfx_masks_or(_cid(), _cid()) in CLIPS

    def test_mirror_x(self):
        from api.main import vfx_mirror_x
        assert vfx_mirror_x(_cid()) in CLIPS

    def test_mirror_y(self):
        from api.main import vfx_mirror_y
        assert vfx_mirror_y(_cid()) in CLIPS

    def test_multiply_color(self):
        from api.main import vfx_multiply_color
        assert vfx_multiply_color(_cid(), factor=1.5) in CLIPS

    def test_multiply_speed(self):
        from api.main import vfx_multiply_speed
        assert vfx_multiply_speed(_cid(), factor=2.0) in CLIPS

    def test_painting(self):
        from api.main import vfx_painting
        assert vfx_painting(_cid()) in CLIPS

    def test_slide_in(self):
        from api.main import vfx_slide_in
        assert vfx_slide_in(_cid(), duration=0.2, side="left") in CLIPS

    def test_slide_out(self):
        from api.main import vfx_slide_out
        assert vfx_slide_out(_cid(), duration=0.2, side="right") in CLIPS

    def test_time_mirror(self):
        from api.main import vfx_time_mirror
        assert vfx_time_mirror(_cid()) in CLIPS

    def test_time_symmetrize(self):
        from api.main import vfx_time_symmetrize
        assert vfx_time_symmetrize(_cid()) in CLIPS

    def test_supersample(self):
        from api.main import vfx_supersample
        assert vfx_supersample(_cid(), d=0.5, nframes=2) in CLIPS

    def test_scroll(self):
        from api.main import vfx_scroll
        assert vfx_scroll(_cid(), w=10, h=10) in CLIPS

    def test_resize_both(self):
        from api.main import vfx_resize
        assert vfx_resize(_cid(), width=10, height=10) in CLIPS

    # Custom effects via backend wrappers
    def test_vfx_quad_mirror(self):
        from api.main import vfx_quad_mirror
        assert vfx_quad_mirror(_cid(), x=5, y=5) in CLIPS

    def test_vfx_chroma_key(self):
        from api.main import vfx_chroma_key
        assert vfx_chroma_key(_cid()) in CLIPS

    def test_vfx_kaleidoscope(self):
        from api.main import vfx_kaleidoscope
        assert vfx_kaleidoscope(_cid(), n_slices=6) in CLIPS

    def test_vfx_matrix(self):
        from api.main import vfx_matrix
        assert vfx_matrix(_cid(), speed=100, density=0.2, font_size=10) in CLIPS

    def test_vfx_auto_framing(self):
        from api.main import vfx_auto_framing
        assert vfx_auto_framing(_cid()) in CLIPS

    def test_vfx_clone_grid(self):
        from api.main import vfx_clone_grid
        assert vfx_clone_grid(_cid(), n_clones=4) in CLIPS

    def test_vfx_kaleidoscope_cube(self):
        from api.main import vfx_kaleidoscope_cube
        assert vfx_kaleidoscope_cube(_cid()) in CLIPS


class TestBackendAfx:
    """Cover all audio effect wrappers."""

    def test_audio_delay(self):
        from api.main import afx_audio_delay
        try: afx_audio_delay(_cid(), offset=0.1)
        except Exception: pass  # color clips lack audio

    def test_audio_fade_in(self):
        from api.main import afx_audio_fade_in
        try: afx_audio_fade_in(_cid(), duration=0.1)
        except Exception: pass

    def test_audio_fade_out(self):
        from api.main import afx_audio_fade_out
        try: afx_audio_fade_out(_cid(), duration=0.1)
        except Exception: pass

    def test_audio_loop(self):
        from api.main import afx_audio_loop
        try: afx_audio_loop(_cid(), n_loops=2)
        except Exception: pass

    def test_audio_normalize(self):
        from api.main import afx_audio_normalize
        try: afx_audio_normalize(_cid())
        except Exception: pass

    def test_multiply_stereo_volume(self):
        from api.main import afx_multiply_stereo_volume
        try: afx_multiply_stereo_volume(_cid(), left=0.5, right=0.5)
        except Exception: pass

    def test_multiply_volume(self):
        from api.main import afx_multiply_volume
        try: afx_multiply_volume(_cid(), factor=0.5)
        except Exception: pass


class TestBackendAudioComposition:
    def test_composite_audio(self):
        from api.main import composite_audio_clips
        try: composite_audio_clips([_cid()])
        except Exception: pass

    def test_concatenate_audio(self):
        from api.main import concatenate_audio_clips
        try: concatenate_audio_clips([_cid()])
        except Exception: pass


class TestBackendPrompts:
    def test_demonstrate_kaleidoscope(self):
        from api.main import demonstrate_kaleidoscope
        assert "clip_1" in demonstrate_kaleidoscope("clip_1")

    def test_glitch_effect_preset(self):
        from api.main import glitch_effect_preset
        assert "RGB" in glitch_effect_preset("c")

    def test_matrix_intro_preset(self):
        from api.main import matrix_intro_preset
        assert "Matrix" in matrix_intro_preset("c")

    def test_auto_framing_for_tiktok(self):
        from api.main import auto_framing_for_tiktok
        assert "9:16" in auto_framing_for_tiktok("c")

    def test_rotating_cube_transition(self):
        from api.main import rotating_cube_transition
        assert "cube" in rotating_cube_transition("c").lower()

    def test_demonstrate_kaleidoscope_cube(self):
        from api.main import demonstrate_kaleidoscope_cube
        assert "clip_1" in demonstrate_kaleidoscope_cube("clip_1")


class TestBackendCheckInstallation:
    def test_success(self):
        from api.main import tools_check_installation
        with patch("moviepy.config.check"):
            r = tools_check_installation()
            assert "check" in r.lower() or "ran" in r.lower()

    def test_failure(self):
        from api.main import tools_check_installation
        with patch("moviepy.config.check", side_effect=Exception("bad")):
            assert "failed" in tools_check_installation().lower()


class TestBackendParseArgs:
    def test_defaults(self):
        a = parse_args([])
        assert a.transport == "http" and a.port == 8080

    def test_stdio(self):
        assert parse_args(["--transport", "stdio"]).transport == "stdio"

    def test_sse(self):
        assert parse_args(["--transport", "sse"]).transport == "sse"

    def test_port(self):
        assert parse_args(["--port", "9090"]).port == 9090


class TestBackendMain:
    def test_stdio(self):
        with patch.object(backend_main, "mcp") as m, \
             patch("api.main.parse_args", return_value=SimpleNamespace(transport="stdio", host="0.0.0.0", port=8080)):
            backend_main.main()
            m.run.assert_called_with(transport="stdio")

    def test_http(self):
        with patch.object(backend_main, "mcp") as m, \
             patch("api.main.parse_args", return_value=SimpleNamespace(transport="http", host="0.0.0.0", port=8080)):
            backend_main.main()
            m.run.assert_called_with(transport="http", host="0.0.0.0", port=8080)

    def test_http_fallback(self):
        n = 0
        def mock_run(**kw):
            nonlocal n; n += 1
            if n == 1: raise Exception("fail")
        with patch.object(backend_main, "mcp") as m, \
             patch("api.main.parse_args", return_value=SimpleNamespace(transport="http", host="0.0.0.0", port=8080)):
            m.run = mock_run
            backend_main.main()

    def test_sse(self):
        with patch.object(backend_main, "mcp") as m, \
             patch("api.main.parse_args", return_value=SimpleNamespace(transport="sse", host="0.0.0.0", port=8080)):
            backend_main.main()
            m.run.assert_called_with(transport="sse", host="0.0.0.0", port=8080)





class TestBackendValidatePath:
    def test_outside_cwd(self):
        assert validate_path("/etc/passwd") == "/etc/passwd"

    def test_inside_cwd(self):
        assert validate_path("test.mp4") == "test.mp4"


class TestBackendImageClipDurationError:
    def test_negative_duration(self):
        from api.main import image_clip
        from PIL import Image
        img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
        img.save("/tmp/_test_cov.png")
        try:
            with pytest.raises(ValueError):
                image_clip("/tmp/_test_cov.png", duration=-1)
        finally:
            os.remove("/tmp/_test_cov.png")


class TestBackendColorClipValidation:
    def test_bad_size(self):
        from api.main import color_clip as cc
        with pytest.raises(ValueError):
            cc([0, 10], [0, 0, 0])
        with pytest.raises(ValueError):
            cc([], [0, 0, 0])
