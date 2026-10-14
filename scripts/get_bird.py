"""Download BIRD Mini-Dev (databases + questions) into data/minidev/.

    python scripts/get_bird.py
"""
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
DEST = Path("data/minidev")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    archive = DEST / "minidev.zip"
    part = DEST / "minidev.zip.part"
    if not archive.exists():
        # Resume an interrupted download instead of starting over (saves data).
        done = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(URL, headers={"Range": f"bytes={done}-"} if done else {})
        print(f"Downloading {URL} (about 1.3 GB)" + (f", resuming at {done >> 20} MB" if done else "") + "...")
        with urllib.request.urlopen(req) as r:
            mode = "ab" if done and r.status == 206 else "wb"
            with part.open(mode) as f:
                shutil.copyfileobj(r, f, length=1 << 20)
        part.rename(archive)
    print("Unzipping...")
    with zipfile.ZipFile(archive) as z:
        z.extractall(DEST)
    found = list(DEST.rglob("mini_dev_sqlite.json"))
    dbs = [p for p in DEST.rglob("dev_databases") if p.is_dir()]
    if not found or not dbs:
        sys.exit("Unzipped, but couldn't find mini_dev_sqlite.json / dev_databases. Check data/minidev/.")
    print(f"Questions: {found[0]}\nDatabases: {dbs[0]}")
    inner = next(dbs[0].iterdir(), None)
    if inner and inner.suffix == ".zip":
        print("Note: databases are zipped individually; unzip them in place.")


if __name__ == "__main__":
    main()
