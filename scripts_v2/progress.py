"""Resumable-run ledger: PROGRESS.json in the run folder.

Each sub-job has a key (e.g. "v42/roll24/2025/cs33.75") and a status:
started | done, with SHA-256 of its output files when done. Writes are atomic
(write <file>.tmp, then os.replace), so a power cut never leaves a half-written
ledger. A sub-job counts as done only if it is marked done AND every recorded
output still exists with the recorded hash; anything else is redone.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Ledger:
    def __init__(self, out_dir):
        self.path = Path(out_dir) / "PROGRESS.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def _flush(self):
        atomic_write_text(self.path, json.dumps(self.data, indent=1, sort_keys=True))

    def is_done(self, key) -> bool:
        e = self.data.get(key)
        if not e or e.get("status") != "done":
            return False
        for p, h in e.get("outputs", {}).items():
            if not Path(p).exists() or sha256(p) != h:
                return False
        return True

    def started(self, key):
        self.data[key] = dict(status="started", t=time.strftime("%Y-%m-%d %H:%M:%S"))
        self._flush()

    def done(self, key, outputs):
        self.data[key] = dict(status="done", t=time.strftime("%Y-%m-%d %H:%M:%S"),
                              outputs={str(p): sha256(p) for p in outputs})
        self._flush()


def _cli(argv=None):
    """python -m scripts_v2.progress --out DIR is-done KEY            (exit 0 = done)
       python -m scripts_v2.progress --out DIR started KEY
       python -m scripts_v2.progress --out DIR done KEY FILE [FILE ...]
       python -m scripts_v2.progress --out DIR status"""
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("cmd", choices=("is-done", "started", "done", "status"))
    ap.add_argument("key", nargs="?")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args(argv)
    led = Ledger(a.out)
    if a.cmd == "is-done":
        sys.exit(0 if led.is_done(a.key) else 1)
    if a.cmd == "started":
        led.started(a.key)
    elif a.cmd == "done":
        missing = [f for f in a.files if not Path(f).exists()]
        if missing:
            print("missing outputs, not marked done:", missing)
            sys.exit(2)
        led.done(a.key, [Path(f).resolve() for f in a.files])
    else:
        for k in sorted(led.data):
            ok = "done" if led.is_done(k) else led.data[k]["status"] + " (redo)"
            print(f"{k:45s} {ok}")


if __name__ == "__main__":
    _cli()
