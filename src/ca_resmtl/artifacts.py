"""Local run manifests and source fingerprints."""

import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def save_run_manifest(output_dir, args, split_info, extra=None, filename="run_manifest.json"):
    """Record source hashes and versions without publishing machine identity."""
    package = Path(__file__).parent
    files = {
        path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    versions = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "openpyxl", "tqdm"):
        versions[name] = importlib.metadata.version(name)
    path = Path(output_dir) / filename
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest = {
        **previous,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": digest,
        "source_files_sha256": files,
        "python_version": platform.python_version(),
        "package_versions": versions,
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if not key.startswith("_")
        },
        "split_info": {**previous.get("split_info", {}), **split_info},
    }
    if extra:
        extra = dict(extra)
        if isinstance(extra.get("y_meta"), dict):
            extra["y_meta"] = {
                key: value for key, value in extra["y_meta"].items() if key != "valid_mask"
            }
        manifest.update(extra)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
