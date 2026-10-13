"""Temporary authenticated transfer over an already-authorized remote session.

The server exposes only a specified run's artifacts, and accepts deployment of
an explicit code allowlist. It never starts/stops training or deletes artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote
from urllib.request import Request, urlopen

DEPLOY_FILES = (
    "scripts/train_relative_image_rollout.py", "scripts/transfer_training_artifacts.py",
    "src/molai/models/relative_rollout.py", "src/molai/models/losses.py",
    "configs/relative_image_rollout.yaml", "configs/model.yaml",
    "configs/navigation_ustat.yaml",
    "scripts/prepare_spectrum_field_dataset.py", "scripts/audit_spectrum_inputs.py",
    "src/molai/data/spectrum_metadata.py", "tests/test_spectrum_metadata.py",
    "src/molai/models/condition.py", "tests/test_metadata_cross_attention.py",
    "scripts/train_absolute_cloud.py", "scripts/train_hidden_rollout.py",
    "scripts/train_cloud.py",
    "docs/competition_ms_inputs.md",
    "src/molai/models/attention.py", "src/molai/models/cloud.py",
    "tests/test_refine_residual_gates.py",
    "tests/test_current_mean_global_vp_loss.py", "tests/test_navigation_training_setup.py",
    "tests/test_vp_cloud_geometry_loss.py",
)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def serve(root, workspace, address, port, token):
    root, workspace = root.resolve(), workspace.resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def authorized(self):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                self.send_error(403)
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == "/manifest":
                rows = [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size,
                         "sha256": sha256(p)} for p in sorted(root.rglob("*"))
                        if p.is_file() and p.resolve().is_relative_to(root)]
                data = json.dumps(rows).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if not self.path.startswith("/file/"):
                self.send_error(404)
                return
            path = (root / unquote(self.path[6:])).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)

        def do_PUT(self):
            if not self.authorized():
                return
            relative = unquote(self.path.removeprefix("/deploy/"))
            if not self.path.startswith("/deploy/") or relative not in DEPLOY_FILES:
                self.send_error(403)
                return
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 2 * 1024 * 1024:
                self.send_error(413)
                return
            path = (workspace / relative).resolve()
            if not path.is_relative_to(workspace):
                self.send_error(403)
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            # Save the pre-deploy source alongside the stopped run for recovery.
            backup = root / "code_before_navigation" / relative
            if path.is_file() and not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
            data = self.rfile.read(size)
            if len(data) != size:
                self.send_error(400)
                return
            temporary = path.with_name(path.name + ".navigation-upload")
            temporary.write_bytes(data)
            temporary.replace(path)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    print(f"Serving scoped artifact transfer on {address}:{port}", flush=True)
    ThreadingHTTPServer((address, port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("serve", "download", "deploy"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--address", default="10.244.234.189")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    token = os.environ["MOLAI_TRANSFER_TOKEN"]
    if args.mode == "serve":
        serve(args.root, args.workspace, args.address, args.port, token)
        return
    base = f"http://{args.address}:{args.port}"
    headers = {"Authorization": f"Bearer {token}"}
    if args.mode == "deploy":
        for relative in DEPLOY_FILES:
            data = (args.workspace / relative).read_bytes()
            request = Request(base + "/deploy/" + quote(relative), data=data, headers=headers, method="PUT")
            with urlopen(request, timeout=60) as response:
                response.read()
            print(f"Deployed {relative}", flush=True)
        return
    with urlopen(Request(base + "/manifest", headers=headers), timeout=120) as response:
        manifest = json.load(response)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        destination = (root / row["path"]).resolve()
        if not destination.is_relative_to(root):
            raise ValueError("manifest path escapes destination")
        if destination.is_file() and destination.stat().st_size == row["size"] and sha256(destination) == row["sha256"]:
            return f"Verified existing {row['path']}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".download")
        with (
            urlopen(Request(base + "/file/" + quote(row["path"]), headers=headers), timeout=120) as response,
            temporary.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        if temporary.stat().st_size != row["size"] or sha256(temporary) != row["sha256"]:
            raise RuntimeError(f"checksum mismatch for {row['path']}")
        temporary.replace(destination)
        return f"Downloaded and SHA256 verified {row['path']}"

    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(fetch, manifest):
            print(result, flush=True)
    (root / "remote-transfer-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified {len(manifest)} files, {sum(row['size'] for row in manifest):,} bytes", flush=True)


if __name__ == "__main__":
    main()
