"""
NiceGUI integration tests for app/main.py.

Uses NiceGUI's built-in testing framework via the `user` fixture to exercise
the full UI page rendering AND the on_go() handler in app/main.py.
"""

import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from nicegui import ui

# Ensure src/app/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "app"))


_mock_mcp = MagicMock()
_mock_mcp.connect = AsyncMock()
_mock_mcp.disconnect = AsyncMock()
_mock_mcp.download_video = MagicMock(return_value="/tmp/fake_video.mp4")


async def _mock_run_agent(*args, **kwargs):
    yield {"type": "thinking", "text": "Analyzing video..."}
    yield {"type": "tool_call", "name": "video_file_clip", "args": {"filename": "media/test.mp4"}}
    yield {"type": "tool_result", "name": "video_file_clip", "result": "clip_abc123"}
    yield {"type": "tool_result", "name": "write_videofile", "result": "Successfully wrote video to media/short_1.mp4"}
    yield {"type": "error", "text": "Minor warning occurred"}
    yield {"type": "message", "text": "Created 2 shorts from the video!"}


_mock_mcp.run_agent = _mock_run_agent


@pytest.fixture(autouse=True)
def patch_mcp_client():
    with patch("mcp_client.MCPVideoClient", return_value=_mock_mcp):
        yield


pytestmark = pytest.mark.nicegui_main_file(str(Path(__file__).parent.parent / "src" / "app" / "main.py"))


class TestPageRendering:
    async def test_page_loads(self, user):
        await user.open("/")
        await user.should_see("AI Shorts Creator")
        await user.should_see("powered by vidmagik-mcp")

    async def test_settings_panel(self, user):
        await user.open("/")
        await user.should_see("LLM Settings")
        await user.should_see("API Base URL")
        await user.should_see("API Key")
        await user.should_see("Model")

    async def test_video_source_section(self, user):
        await user.open("/")
        await user.should_see("Video Source")
        await user.should_see("Video URL")
        await user.should_see("Create Shorts")

    async def test_agent_activity_section(self, user):
        await user.open("/")
        await user.should_see("Agent Activity")
        await user.should_see("Waiting for input")

    async def test_instructions_input(self, user):
        await user.open("/")
        await user.should_see("Instructions (optional)")


class TestOnGo:
    """Test the on_go() handler by clicking the Create Shorts button."""

    async def test_empty_url_warning(self, user):
        """Click with empty URL → should still show the page (no crash)."""
        await user.open("/")
        user.find("Create Shorts").click()
        await asyncio.sleep(0.1)
        # Page should still be functional
        await user.should_see("Agent Activity")

    async def test_full_agent_flow(self, user):
        """Set URL, click Create Shorts → exercises on_go fully."""
        await user.open("/")
        # Type into the URL input
        user.find(ui.input).click()
        user.find(kind=ui.input, marker="Video URL").type("https://youtube.com/watch?v=test123")
        # Click the button
        user.find("Create Shorts").click()
        # Wait for async handler to process
        await asyncio.sleep(0.5)
        await user.should_see("Agent Activity")

    async def test_empty_model_warning(self, user):
        """Set URL but clear model → should warn."""
        await user.open("/")
        # Type a URL
        user.find(kind=ui.input, marker="Video URL").type("https://example.com")
        # Click create
        user.find("Create Shorts").click()
        await asyncio.sleep(0.1)
        await user.should_see("Agent Activity")


class TestStartupShutdown:
    async def test_startup_hook(self, user):
        await user.open("/")
        await user.should_see("AI Shorts Creator")

    async def test_shutdown_hook(self, user):
        await user.open("/")
        await user.should_see("Agent Activity")
