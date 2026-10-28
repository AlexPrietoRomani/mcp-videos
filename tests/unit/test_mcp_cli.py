"""Unit tests for the FastMCP tool server and the Typer CLI contract."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from typer.testing import CliRunner

import mcp_videos.cli as cli
import mcp_videos.mcp_server as server
from mcp_videos.cli import SystemCheckResult

EXPECTED_TOOLS = {
    "create_project",
    "list_voices",
    "register_voice",
    "render_project_video",
    "save_and_validate_scene",
    "synthesize_dialogue",
}


class FakeOrchestrator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_on: str | None = None

    def _record(self, operation: str, **payload: Any) -> None:
        if self.fail_on == operation:
            raise RuntimeError(f"fallo simulado en {operation}")
        self.calls.append({"operation": operation, **payload})

    def register_voice(
        self,
        voice_id: str,
        name: str | None = None,
        *,
        engine: str = "kokoro",
        mode: str = "preset",
        sample_wav_path: str | None = None,
        language: str = "es",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._record(
            "register_voice",
            voice_id=voice_id,
            name=name,
            engine=engine,
            mode=mode,
            sample_wav_path=sample_wav_path,
            language=language,
            config=config,
        )
        return {
            "status": "ready",
            "voice_id": voice_id,
            "name": name,
            "engine": engine,
            "mode": mode,
            "language": language,
            "sample_rate": 24000,
        }

    def list_voices(self, language: str | None = None) -> list[dict[str, Any]]:
        self._record("list_voices", language=language)
        return [
            {
                "voice_id": "preset-narrador-es",
                "name": "Narrador",
                "engine": "kokoro",
                "mode": "preset",
                "language": language or "es",
            }
        ]

    def create_project(
        self,
        project_id: str,
        title: str,
        *,
        width: int = 1080,
        height: int = 1920,
        fps: int = 60,
        description: str = "",
    ) -> dict[str, Any]:
        self._record(
            "create_project",
            project_id=project_id,
            title=title,
            width=width,
            height=height,
            fps=fps,
            description=description,
        )
        return {
            "project_id": project_id,
            "title": title,
            "workspace_dir": f"/workspaces/{project_id}",
            "status": "draft",
            "width": width,
            "height": height,
            "fps": fps,
        }

    def synthesize_and_align(
        self,
        project_id: str,
        turns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._record("synthesize_and_align", project_id=project_id, turns=turns)
        return {
            "project_id": project_id,
            "status": "audio_ready",
            "total_duration_ms": 2400,
            "master_audio_path": "audio/dialogue_master.wav",
            "lines_count": len(turns),
            "words_count": 12,
            "lines": [],
            "words": [],
        }

    def save_and_validate_scene(
        self,
        project_id: str,
        scene_index: int,
        title: str,
        start_ms: int,
        end_ms: int,
        *,
        html_content: str | None = None,
        prompt: str = "",
        provider: str | None = None,
    ) -> dict[str, Any]:
        self._record(
            "save_and_validate_scene",
            project_id=project_id,
            scene_index=scene_index,
            title=title,
            start_ms=start_ms,
            end_ms=end_ms,
            html_content=html_content,
            prompt=prompt,
            provider=provider,
        )
        return {
            "project_id": project_id,
            "scene_index": scene_index,
            "title": title,
            "validation_status": "valid",
            "validation_errors": [],
            "console_errors": [],
            "scenes_ready": True,
            "project_status": "scenes_ready",
        }

    def render_project(
        self,
        project_id: str,
        *,
        burn_subtitles: bool = True,
    ) -> dict[str, Any]:
        self._record(
            "render_project",
            project_id=project_id,
            burn_subtitles=burn_subtitles,
        )
        return {
            "job_id": "job-1",
            "project_id": project_id,
            "status": "completed",
            "project_status": "rendered",
            "output_mp4_path": f"/workspaces/{project_id}/output/final.mp4",
            "output_mp4_relative_path": "output/final.mp4",
            "fps": 60,
            "total_frames": 120,
            "rendered_frames": 120,
            "duration_ms": 2400,
            "subtitles_path": f"/workspaces/{project_id}/output/final.ass",
        }

    def build_demo_short(self, project_id: str, **kwargs: Any) -> dict[str, Any]:
        self._record("build_demo_short", project_id=project_id, **kwargs)
        return {
            "project_id": project_id,
            "title": kwargs.get("title", "Demo"),
            "project": {"status": "scenes_ready"},
            "workspace_dir": f"/workspaces/{project_id}",
            "voices": [],
            "synthesis": {"total_duration_ms": 2400},
            "scenes": [{"validation_status": "valid"}, {"validation_status": "valid"}],
            "render": None,
            "final_video_path": f"/workspaces/{project_id}/output/final.mp4",
        }


def mcp_payload(result: Any) -> Any:
    payload = result.structured_content
    if payload is None:
        return json.loads(result.content[0].text)
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    return asyncio.run(server.mcp.call_tool(name, arguments))


@pytest.fixture(autouse=True)
def reset_mcp_orchestrator() -> None:
    server.set_orchestrator(None)
    yield
    server.set_orchestrator(None)


@pytest.fixture
def fake() -> FakeOrchestrator:
    service = FakeOrchestrator()
    server.set_orchestrator(service)
    return service


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestMcpServer:
    def test_registers_exactly_the_documented_tools(self) -> None:
        tools = asyncio.run(server.mcp.list_tools())

        assert {tool.name for tool in tools} == EXPECTED_TOOLS

    def test_register_voice_tool_returns_ready_payload(self, fake: FakeOrchestrator) -> None:
        result = call_tool(
            "register_voice",
            {
                "voice_id": "voz-clon",
                "name": "Clon de prueba",
                "mode": "clone",
                "sample_wav_path": "/samples/clon.wav",
                "config": {"ref_text": "muestra"},
            },
        )

        payload = mcp_payload(result)
        assert result.is_error is False
        assert payload["status"] == "ready"
        assert payload["voice_id"] == "voz-clon"
        assert fake.calls[0]["mode"] == "clone"
        assert fake.calls[0]["config"] == {"ref_text": "muestra"}

    def test_list_voices_tool_returns_catalog(self, fake: FakeOrchestrator) -> None:
        result = call_tool("list_voices", {"language": "es"})

        payload = mcp_payload(result)
        assert result.is_error is False
        assert payload[0]["voice_id"] == "preset-narrador-es"
        assert fake.calls[0]["language"] == "es"

    def test_create_and_synthesize_tools_forward_arguments(
        self,
        fake: FakeOrchestrator,
    ) -> None:
        created = mcp_payload(
            call_tool(
                "create_project",
                {"project_id": "short-mcp", "title": "Antena", "width": 360, "height": 640},
            )
        )
        turns = [
            {
                "voice_id": "preset-narrador-es",
                "speaker_label": "Narrador",
                "text_content": "Hola",
                "pause_after_ms": 0,
            }
        ]
        synthesis = mcp_payload(
            call_tool(
                "synthesize_dialogue",
                {"project_id": "short-mcp", "lines": turns},
            )
        )

        assert created["project_id"] == "short-mcp"
        assert created["width"] == 360
        assert synthesis["status"] == "audio_ready"
        assert synthesis["lines_count"] == 1
        assert fake.calls[1]["turns"] == turns

    def test_save_scene_and_render_tools_forward_arguments(
        self,
        fake: FakeOrchestrator,
    ) -> None:
        scene = mcp_payload(
            call_tool(
                "save_and_validate_scene",
                {
                    "project_id": "short-mcp",
                    "scene_index": 0,
                    "title": "Hook",
                    "start_ms": 0,
                    "end_ms": 1200,
                    "html_content": "<html></html>",
                    "prompt": "plantilla",
                },
            )
        )
        rendered = mcp_payload(
            call_tool(
                "render_project_video",
                {"project_id": "short-mcp", "burn_subtitles": False},
            )
        )

        assert scene["validation_status"] == "valid"
        assert fake.calls[0]["html_content"] == "<html></html>"
        assert rendered["status"] == "completed"
        assert fake.calls[1]["burn_subtitles"] is False

    def test_tool_errors_are_returned_as_structured_payloads(
        self,
        fake: FakeOrchestrator,
    ) -> None:
        fake.fail_on = "create_project"
        payload = mcp_payload(
            call_tool("create_project", {"project_id": "short-mcp", "title": "Antena"})
        )

        assert payload["status"] == "error"
        assert payload["error_type"] == "RuntimeError"
        assert "fallo simulado" in payload["error"]

    def test_list_voices_error_is_returned_as_structured_payload(
        self,
        fake: FakeOrchestrator,
    ) -> None:
        fake.fail_on = "list_voices"
        payload = mcp_payload(call_tool("list_voices", {}))

        assert payload[0]["status"] == "error"
        assert payload[0]["error_type"] == "RuntimeError"


class TestCli:
    def test_doctor_json_reports_healthy_system(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            cli,
            "check_system_dependencies",
            lambda: SystemCheckResult(
                ffmpeg_path="/usr/bin/ffmpeg",
                ffprobe_path="/usr/bin/ffprobe",
                chromium_path="/usr/bin/chromium",
            ),
        )

        result = runner.invoke(cli.app, ["doctor", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ok"
        assert payload["healthy"] is True
        assert payload["ffmpeg"] == "/usr/bin/ffmpeg"

    def test_doctor_exits_non_zero_when_dependencies_are_missing(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            cli,
            "check_system_dependencies",
            lambda: SystemCheckResult(None, None, None),
        )

        result = runner.invoke(cli.app, ["doctor"])

        assert result.exit_code == 1
        assert "MISSING" in result.output
        assert "status: ERROR" in result.output

    def test_voices_command_uses_orchestrator(
        self,
        runner: CliRunner,
        fake: FakeOrchestrator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(cli, "get_orchestrator", lambda: fake)

        result = runner.invoke(cli.app, ["voices", "--language", "es", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload[0]["voice_id"] == "preset-narrador-es"
        assert fake.calls[0]["language"] == "es"

    def test_register_voice_parses_json_config(
        self,
        runner: CliRunner,
        fake: FakeOrchestrator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(cli, "get_orchestrator", lambda: fake)

        result = runner.invoke(
            cli.app,
            [
                "register-voice",
                "--voice-id",
                "voz-narrador",
                "--name",
                "Narrador",
                "--config",
                '{"temperature": 0.2}',
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "ready"
        assert fake.calls[0]["config"] == {"temperature": 0.2}

    def test_create_project_command(
        self,
        runner: CliRunner,
        fake: FakeOrchestrator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(cli, "get_orchestrator", lambda: fake)

        result = runner.invoke(
            cli.app,
            [
                "create-project",
                "--project-id",
                "short-cli",
                "--title",
                "Antena",
                "--width",
                "360",
                "--height",
                "640",
                "--fps",
                "24",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["project_id"] == "short-cli"
        assert fake.calls[0]["width"] == 360
        assert fake.calls[0]["fps"] == 24

    def test_synthesize_reads_turns_from_json_file(
        self,
        runner: CliRunner,
        fake: FakeOrchestrator,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(cli, "get_orchestrator", lambda: fake)
        script = tmp_path / "script.json"
        turns = [
            {
                "voice_id": "preset-narrador-es",
                "speaker_label": "Narrador",
                "text_content": "Hola",
            }
        ]
        script.write_text(json.dumps(turns), encoding="utf-8")

        result = runner.invoke(
            cli.app,
            [
                "synthesize",
                "--project-id",
                "short-cli",
                "--script-file",
                str(script),
                "--json",
            ],
        )

        assert result.exit_code == 0
        assert fake.calls[0]["turns"] == turns

    def test_save_scene_reads_html_file(
        self,
        runner: CliRunner,
        fake: FakeOrchestrator,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(cli, "get_orchestrator", lambda: fake)
        html_file = tmp_path / "scene.html"
        html_file.write_text("<html><body>Hola</body></html>", encoding="utf-8")

        result = runner.invoke(
            cli.app,
            [
                "save-scene",
                "--project-id",
                "short-cli",
                "--scene-index",
                "0",
                "--start-ms",
                "0",
                "--end-ms",
                "1200",
                "--title",
                "Hook",
                "--html-file",
                str(html_file),
                "--prompt",
                "plantilla",
                "--json",
            ],
        )

        assert result.exit_code == 0
        assert fake.calls[0]["html_content"] == "<html><body>Hola</body></html>"
        assert fake.calls[0]["prompt"] == "plantilla"

    def test_render_command(
        self,
        runner: CliRunner,
        fake: FakeOrchestrator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(cli, "get_orchestrator", lambda: fake)

        result = runner.invoke(
            cli.app,
            ["render", "--project-id", "short-cli", "--no-burn-subtitles", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status"] == "completed"
        assert fake.calls[0]["burn_subtitles"] is False

    def test_build_demo_command(
        self,
        runner: CliRunner,
        fake: FakeOrchestrator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(cli, "get_orchestrator", lambda: fake)

        result = runner.invoke(
            cli.app,
            [
                "build-demo",
                "--project-id",
                "short-demo-cli",
                "--width",
                "360",
                "--height",
                "640",
                "--fps",
                "24",
                "--no-render",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["project_id"] == "short-demo-cli"
        assert fake.calls[0]["width"] == 360
        assert fake.calls[0]["render"] is False

    def test_preview_command_imports_server_lazily(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        created: list[str] = []
        server_module = types.ModuleType("mcp_videos.preview.server")
        server_module.create_preview_app = lambda: created.append("app")
        monkeypatch.setitem(sys.modules, "mcp_videos.preview.server", server_module)
        launched: dict[str, Any] = {}

        def fake_run(preview_app: object, *, host: str, port: int) -> None:
            launched.update({"app": preview_app, "host": host, "port": port})

        monkeypatch.setattr(uvicorn, "run", fake_run)

        result = runner.invoke(cli.app, ["preview", "--host", "0.0.0.0", "--port", "9091"])

        assert result.exit_code == 0
        assert created == ["app"]
        assert launched == {"app": None, "host": "0.0.0.0", "port": 9091}

    def test_command_errors_exit_non_zero(
        self,
        runner: CliRunner,
        fake: FakeOrchestrator,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake.fail_on = "list_voices"
        monkeypatch.setattr(cli, "get_orchestrator", lambda: fake)

        result = runner.invoke(cli.app, ["voices"])

        assert result.exit_code == 1
        assert "Error: fallo simulado en list_voices" in result.output
