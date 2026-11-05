"""SQLite storage: prompts, immutable versions and evaluation runs."""
from __future__ import annotations

import difflib
import json
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS prompts(id INTEGER PRIMARY KEY, name TEXT UNIQUE, created TEXT);
CREATE TABLE IF NOT EXISTS versions(id INTEGER PRIMARY KEY, prompt_id INTEGER, version INTEGER,
    variant TEXT, text TEXT, created TEXT, UNIQUE(prompt_id, version));
CREATE TABLE IF NOT EXISTS runs(id INTEGER PRIMARY KEY, version_id INTEGER, model TEXT,
    output TEXT, scores TEXT, created TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str = "promptforge.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def _prompt_id(self, name: str) -> int:
        self.db.execute("INSERT OR IGNORE INTO prompts(name, created) VALUES(?,?)", (name, _now()))
        return self.db.execute("SELECT id FROM prompts WHERE name=?", (name,)).fetchone()["id"]

    def add_version(self, name: str, text: str, variant: str = "manual") -> int:
        pid = self._prompt_id(name)
        last = self.db.execute("SELECT MAX(version) m FROM versions WHERE prompt_id=?", (pid,)).fetchone()["m"] or 0
        cur = self.db.execute(
            "INSERT INTO versions(prompt_id, version, variant, text, created) VALUES(?,?,?,?,?)",
            (pid, last + 1, variant, text, _now()))
        self.db.commit()
        return cur.lastrowid

    def add_run(self, version_id: int, model: str, output: str, scores: dict) -> None:
        self.db.execute("INSERT INTO runs(version_id, model, output, scores, created) VALUES(?,?,?,?,?)",
                        (version_id, model, output, json.dumps(scores), _now()))
        self.db.commit()

    def list_prompts(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT p.name, COUNT(v.id) versions FROM prompts p LEFT JOIN versions v ON v.prompt_id=p.id "
            "GROUP BY p.id ORDER BY p.id DESC").fetchall()
        return [dict(r) for r in rows]

    def history(self, name: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT v.version, v.variant, v.text, v.created, "
            "(SELECT scores FROM runs r WHERE r.version_id=v.id ORDER BY r.id DESC LIMIT 1) scores "
            "FROM versions v JOIN prompts p ON p.id=v.prompt_id WHERE p.name=? ORDER BY v.version", (name,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["scores"] = json.loads(d["scores"]) if d["scores"] else None
            out.append(d)
        return out

    def diff(self, name: str, a: int, b: int) -> str:
        h = {x["version"]: x["text"] for x in self.history(name)}
        return "\n".join(difflib.unified_diff(h[a].splitlines(), h[b].splitlines(),
                                              f"v{a}", f"v{b}", lineterm=""))
