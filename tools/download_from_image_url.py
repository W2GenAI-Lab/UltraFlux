#!/usr/bin/env python3
"""
Download a single image URL (works for both Pexels and Unsplash).

Unsplash URLs often have no filename extension in the path (e.g. ".../photo-xxxx?..."),
so this script infers a proper extension from the HTTP Content-Type.

Examples:
  python download_from_image_url.py "https://images.unsplash.com/photo-...?...&ixlib=rb-4.1.0"
  python download_from_image_url.py "https://images.pexels.com/photos/4761978/pexels-photo-4761978.jpeg"
  python download_from_image_url.py URL --output_dir ./downloads
  python download_from_image_url.py URL --out my_image.jpg
"""

from __future__ import annotations

import argparse
import os
import re
import ssl
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen


_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_EXT_BY_CONTENT_TYPE = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tif",
}


def _sanitize_filename(name: str) -> str:
    name = name.strip()
    name = os.path.basename(name)
    name = _INVALID_FILENAME_CHARS.sub("_", name)
    name = name.strip(" .")
    return (name or "image")[:200]


def _infer_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    dl = qs.get("dl")
    if dl and dl[0]:
        return _sanitize_filename(unquote(dl[0]))

    last = unquote((parsed.path or "").rstrip("/").split("/")[-1])
    return _sanitize_filename(last or "image")


def _ensure_ext(filename: str, content_type: Optional[str]) -> str:
    if Path(filename).suffix:
        return filename
    ct = (content_type or "").split(";", 1)[0].strip().lower()
    ext = _EXT_BY_CONTENT_TYPE.get(ct, ".jpg")
    return filename + ext


def download(url: str, dst: Path, timeout_s: float, user_agent: str, insecure_tls: bool) -> tuple[Path, str]:
    ctx = None
    if insecure_tls:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=timeout_s, context=ctx) as r:
        content_type = r.headers.get("Content-Type") or ""
        final_name = _ensure_ext(dst.name, content_type)
        if final_name != dst.name:
            dst = dst.with_name(final_name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        try:
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dst)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
    return dst, content_type


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="Image URL (Pexels/Unsplash/etc.)")
    ap.add_argument("--output_dir", type=Path, default=Path("."), help="Directory to save into (default: .)")
    ap.add_argument("--out", type=str, default="", help="Output filename (optional)")
    ap.add_argument("--timeout_s", type=float, default=60.0, help="Request timeout seconds")
    ap.add_argument("--user_agent", type=str, default="Mozilla/5.0 (download_url.py)")
    ap.add_argument("--insecure_tls", action="store_true", help="Disable TLS verification (not recommended)")
    args = ap.parse_args()

    name = _sanitize_filename(args.out) if args.out else _infer_name_from_url(args.url)
    dst = (args.output_dir / name).resolve()

    saved_path, content_type = download(
        args.url,
        dst,
        timeout_s=args.timeout_s,
        user_agent=args.user_agent,
        insecure_tls=args.insecure_tls,
    )
    print(f"saved: {saved_path}")
    print(f"content-type: {content_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())