"""Fetch the pinned Apache-2.0 G1 whole-body get-up policy and motion clip."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import urllib.request


COMMIT = "6dabf86fddc2b7b429b09e74999732fcde3441f9"
BASE_URL = f"https://raw.githubusercontent.com/wbc-mjlab/wbc-g1-deploy/{COMMIT}"
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "third_party" / "wbc_g1_getup"
FILES = {
    "policy.onnx": (
        "config/policy/wbc/params/policy.onnx",
        "63a444122d85d2868045e08ff51c3fe711edbabc2d8c96a1a241d4c59e98bb34",
    ),
    "getup_01.npz": (
        "config/clips/getup_01.npz",
        "32dcbb982351e17bfd8a2bdb8a2affe1d41f538bb0eeea20f2b28c05807ec79f",
    ),
    "LICENSE": (
        "LICENSE",
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_getup_assets() -> Path:
    """Download missing assets and fail closed on any digest mismatch."""

    DESTINATION.mkdir(parents=True, exist_ok=True)
    for filename, (source, expected_digest) in FILES.items():
        target = DESTINATION / filename
        if target.exists() and _sha256(target) == expected_digest:
            continue
        url = f"{BASE_URL}/{source}"
        with tempfile.NamedTemporaryFile(
            prefix=f"{filename}.", dir=DESTINATION, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            print(f"Fetching {url}")
            urllib.request.urlretrieve(url, temporary_path)
            actual_digest = _sha256(temporary_path)
            if actual_digest != expected_digest:
                raise RuntimeError(
                    f"SHA-256 mismatch for {filename}: "
                    f"expected {expected_digest}, received {actual_digest}"
                )
            temporary_path.replace(target)
        finally:
            temporary_path.unlink(missing_ok=True)
    return DESTINATION


if __name__ == "__main__":
    path = ensure_getup_assets()
    print(f"Verified get-up assets in {path}")
