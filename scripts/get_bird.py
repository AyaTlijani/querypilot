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
    if not archive.exists():
        print(f"Downloading {URL} (about 1.3 GB, this can take a while)...")
        with urllib.request.urlopen(URL) as r, archive.open("wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
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
