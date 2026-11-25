"""Security contract for workspace-scoped file and artifact tools."""

import pytest

from core.html_diagram_maker import HTMLDiagramBuilder
from core.tools import AgentToolRegistry
from core.visual_maker import InteractiveVisualMaker
from memory.db import EpisodicMemoryDB


PATH_TOOLS = [
    ("read_file", "path", {}),
    ("write_file", "path", {"content": "blocked"}),
    ("edit_file", "path", {"target": "secret", "replacement": "blocked"}),
    ("list_dir", "path", {}),
    ("json_csv_analyzer", "path", {}),
    ("grep_search", "path", {"query": "secret"}),
    ("ast_lint_checker", "path", {}),
    ("interactive_visual_maker", "filename", {"title": "Boundary"}),
    ("html_diagram_builder", "filename", {"title": "Boundary"}),
    ("generate_image", "filename", {"prompt": "Boundary"}),
    ("generate_image_diffusion", "filename", {"prompt": "Boundary"}),
    ("generate_video_diffusion", "filename", {"prompt": "Boundary"}),
    ("edit_image_rapid", "filename", {"prompt": "Boundary"}),
    ("render_bezier_art", "filename", {}),
    ("export_chat_history", "filename", {}),
]


def test_file_tools_require_and_clear_an_active_workspace(tmp_path):
    registry = AgentToolRegistry(db_path=str(tmp_path / "memory.db"))
    assert registry.get_workspace_dir() is None
    for tool, path_field, base_args in PATH_TOOLS:
        args = dict(base_args)
        args[path_field] = "safe-name.txt"
        assert registry.execute_tool(tool, args) == (
            False,
            "No workspace is selected.",
        )
    assert registry.execute_tool("file_search", {"pattern": "*.txt"}) == (
        False,
        "No workspace is selected.",
    )
    assert registry.execute_tool("run_terminal", {"command": "pwd"}) == (
        False,
        "No workspace is selected.",
    )
    assert registry.execute_tool("python_sandbox", {"code": "print('unsafe')"}) == (
        False,
        "No workspace is selected.",
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert registry.set_workspace_dir(str(workspace))
    assert registry.execute_tool("write_file", {"path": "note.txt", "content": "safe"})[0]
    assert registry.set_workspace_dir(None)
    assert registry.get_workspace_dir() is None
    assert not registry.execute_tool("read_file", {"path": "note.txt"})[0]
    assert registry.set_workspace_dir(str(workspace))
    assert not registry.set_workspace_dir(str(tmp_path / "missing"))
    assert registry.get_workspace_dir() is None


@pytest.mark.parametrize(("tool", "path_field", "base_args"), PATH_TOOLS)
def test_model_paths_reject_absolute_and_parent_escapes(
    tmp_path, tool, path_field, base_args
):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    outside_file = outside / "secret.txt"
    outside_file.write_text("secret", encoding="utf-8")
    registry = AgentToolRegistry(
        db_path=str(workspace / "memory.db"), workspace_dir=str(workspace)
    )

    for attack_path in (str(outside_file), "../outside/secret.txt"):
        args = dict(base_args)
        args[path_field] = attack_path
        ok, _result = registry.execute_tool(tool, args)
        assert not ok


def test_image_edit_source_rejects_workspace_escapes(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.png"
    workspace.mkdir()
    outside.write_bytes(b"outside")
    registry = AgentToolRegistry(workspace_dir=str(workspace))

    for attack_path in (str(outside), "../outside.png"):
        ok, _result = registry.execute_tool(
            "edit_image_rapid",
            {"prompt": "edit", "image_path": attack_path, "filename": "safe.png"},
        )
        assert not ok


def test_outside_symlinks_are_not_read_listed_searched_or_written(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("outside secret", encoding="utf-8")
    try:
        (workspace / "leak.txt").symlink_to(outside / "secret.txt")
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform")

    registry = AgentToolRegistry(workspace_dir=str(workspace))
    assert not registry.execute_tool("read_file", {"path": "leak.txt"})[0]
    assert not registry.execute_tool(
        "write_file", {"path": "leak.txt", "content": "overwrite"}
    )[0]
    assert "leak.txt" not in registry.execute_tool("list_dir", {"path": "."})[1]
    assert "leak.txt" not in registry.execute_tool("file_search", {"pattern": "*.txt"})[1]
    assert "outside secret" not in registry.execute_tool(
        "grep_search", {"query": "outside secret", "path": "."}
    )[1]
    assert (outside / "secret.txt").read_text(encoding="utf-8") == "outside secret"


def test_relative_workspace_operations_and_artifacts_remain_available(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    db_path = workspace / "memory.db"
    EpisodicMemoryDB(db_path).log_interaction(
        "Prompt", "Completion", verified_reward=1
    )
    registry = AgentToolRegistry(db_path=str(db_path), workspace_dir=str(workspace))

    assert registry.execute_tool(
        "write_file", {"path": "notes/note.txt", "content": "before"}
    )[0]
    assert registry.execute_tool("read_file", {"path": "notes/note.txt"})[0]
    assert registry.execute_tool(
        "edit_file",
        {"path": "notes/note.txt", "target": "before", "replacement": "after"},
    )[0]
    assert registry.execute_tool("list_dir", {"path": "notes"})[0]
    assert registry.execute_tool("file_search", {"pattern": "*.txt"})[0]
    (workspace / "data.json").write_text('{"safe": true}', encoding="utf-8")
    (workspace / "valid.py").write_text("def safe():\n    return True\n", encoding="utf-8")
    assert registry.execute_tool("json_csv_analyzer", {"path": "data.json"})[0]
    assert registry.execute_tool(
        "grep_search", {"query": "return True", "path": "."}
    )[0]
    assert registry.execute_tool("ast_lint_checker", {"path": "valid.py"})[0]
    assert registry.execute_tool(
        "interactive_visual_maker", {"title": "Safe", "filename": "safe-visual.html"}
    )[0]
    assert registry.execute_tool(
        "html_diagram_builder", {"title": "Safe", "filename": "safe-diagram.html"}
    )[0]
    assert registry.execute_tool(
        "export_chat_history", {"filename": "safe-export.md"}
    )[0]
    assert registry.execute_tool(
        "generate_image", {"prompt": "Safe", "filename": "safe-image.svg"}
    )[0]
    assert registry.execute_tool(
        "render_bezier_art", {"filename": "safe-bezier.svg"}
    )[0]

    assert (workspace / "notes" / "note.txt").read_text(encoding="utf-8") == "after"
    assert (workspace / "safe-visual.html").is_file()
    assert (workspace / "safe-visual.svg").is_file()
    assert (workspace / "safe-diagram.html").is_file()
    assert (workspace / "safe-export.md").is_file()
    assert (workspace / "safe-image.svg").is_file()
    assert (workspace / "safe-bezier.svg").is_file()


@pytest.mark.parametrize("maker", [InteractiveVisualMaker, HTMLDiagramBuilder])
def test_direct_artifact_builders_require_safe_filenames(tmp_path, maker):
    unscoped = maker()
    result = (
        unscoped.create_visualization(filename="visual.html")
        if maker is InteractiveVisualMaker
        else unscoped.create_diagram(filename="diagram.html")
    )
    assert not result[0] and result[1] == "No workspace is selected."

    scoped = maker(workspace_dir=str(tmp_path))
    result = (
        scoped.create_visualization(filename="../visual.html")
        if maker is InteractiveVisualMaker
        else scoped.create_diagram(filename="../diagram.html")
    )
    assert not result[0]
