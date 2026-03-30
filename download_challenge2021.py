"""
Download PhysioNet 2021 Challenge ECG datasets (open access, no credentials needed).
Downloads: georgia, cpsc_2018, cpsc_2018_extra, ningbo
Skips: chapman_shaoxing, ptb-xl (already have locally)

Usage:
    python download_challenge2021.py
"""
import os
import time
import urllib.request
from pathlib import Path
from html.parser import HTMLParser

BASE_URL  = "https://physionet.org/files/challenge-2021/1.0.3/training/"
SAVE_DIR  = Path("ekg_datasets/challenge2021")
DATASETS  = ["georgia", "cpsc_2018", "cpsc_2018_extra", "ningbo"]

os.chdir(Path(__file__).parent)


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for attr, val in attrs:
                if attr == "href" and val and not val.startswith("?") and val != "../":
                    self.links.append(val)


def list_dir(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            html = r.read().decode()
        p = LinkParser()
        p.feed(html)
        return p.links
    except Exception as e:
        print(f"    ERROR listing {url}: {e}")
        return []


def download_file(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return False  # already downloaded
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"    ERROR {url}: {e}")
        return False


def download_dataset(name):
    base = BASE_URL + name + "/"
    save = SAVE_DIR / name
    print(f"\n[{name}] Listing files...")
    subdirs = list_dir(base)

    # Collect all .mat and .hea file URLs
    files_to_download = []
    for sub in subdirs:
        if sub.endswith("/") and sub != "../":
            sub_url = base + sub
            sub_files = list_dir(sub_url)
            for f in sub_files:
                if f.endswith(".mat") or f.endswith(".hea"):
                    files_to_download.append((sub_url + f, save / sub.rstrip("/") / f))
        elif sub.endswith(".mat") or sub.endswith(".hea"):
            files_to_download.append((base + sub, save / sub))

    total = len(files_to_download)
    print(f"[{name}] Found {total} files to download")

    downloaded = 0
    skipped    = 0
    t0 = time.time()
    for i, (url, dest) in enumerate(files_to_download):
        result = download_file(url, dest)
        if result:
            downloaded += 1
        else:
            skipped += 1
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate
            print(f"  [{name}] {i+1}/{total}  ({rate:.0f} files/s, ~{remaining/60:.0f} min left)",
                  flush=True)

    mat_count = len(list(save.rglob("*.mat")))
    print(f"[{name}] Done: {mat_count} .mat files  (downloaded={downloaded}, skipped={skipped})",
          flush=True)
    return mat_count


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {SAVE_DIR.resolve()}")
    print(f"Datasets: {DATASETS}")

    t_start = time.time()
    for ds in DATASETS:
        download_dataset(ds)

    total_mat = len(list(SAVE_DIR.rglob("*.mat")))
    print(f"\nAll done: {total_mat} total .mat files in {SAVE_DIR}")
    print(f"Total time: {(time.time()-t_start)/60:.0f} min")
