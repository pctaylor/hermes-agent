"""Behavior contracts for the desktop_project model tool."""

import json

from hermes_cli import projects_db as pdb
from tools import project_tools  # noqa: F401 — registers desktop_project
from tools.registry import registry


def test_create_without_path_is_rejected_without_persisting_project():
    db_path = pdb.projects_db_path()
    assert not db_path.exists()

    raw = registry.dispatch("desktop_project", {"action": "create", "name": "Ghost"})
    assert isinstance(raw, str)
    result = json.loads(raw)

    assert result == {"success": False, "error": "path is required for create"}
    assert not db_path.exists()
    with pdb.connect_closing() as conn:
        assert pdb.list_projects(conn, include_archived=True) == []
        assert pdb.get_active_id(conn) is None


def test_create_with_path_still_persists_and_activates_project(tmp_path):
    folder = tmp_path / "workspace"
    folder.mkdir()

    raw = registry.dispatch(
        "desktop_project", {"action": "create", "name": "Workspace", "path": str(folder)})
    assert isinstance(raw, str)
    result = json.loads(raw)

    assert result["success"] is True
    assert result["primary_path"] == str(folder)
    with pdb.connect_closing() as conn:
        projects = pdb.list_projects(conn)
        assert [project.id for project in projects] == [result["id"]]
        assert pdb.get_active_id(conn) == result["id"]
