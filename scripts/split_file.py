import argparse
import os
from pathlib import Path

def split_file(path: str, max_bytes: int, out_prefix: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    with p.open("rb") as f:
        idx = 1
        while True:
            chunk = f.read(max_bytes)
            if not chunk:
                break
            out = Path(f"{out_prefix}{idx:03d}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(chunk)
            print(f"Wrote {out} ({out.stat().st_size} bytes)")
            idx += 1

def main():
    ap = argparse.ArgumentParser(description="Split a large file into chunks for GitHub pushes.")
    ap.add_argument("file", help="Path to input file")
    ap.add_argument("--max-mib", type=int, default=95, help="Max size per part in MiB (default 95)")
    ap.add_argument("--out-prefix", default=None, help="Prefix for output parts (default: <file>.part)")
    args = ap.parse_args()

    max_bytes = args.max_mib * 1024 * 1024
    out_prefix = args.out_prefix or (args.file + ".part")
    split_file(args.file, max_bytes=max_bytes, out_prefix=out_prefix)

if __name__ == "__main__":
    main()
