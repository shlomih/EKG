"""
Download PhysioNet 2021 Challenge ECG datasets (open access, no credentials needed).
Downloads: georgia, cpsc_2018, cpsc_2018_extra, ningbo
Skips: chapman_shaoxing, ptb-xl (already have locally)
Uses parallel threads for fast download.

Usage:
    python download_challenge2021.py
"""
import os
import sys
import time
import urllib.request
from pathlib import Path
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL  = "https://physionet.org/files/challenge-2021/1.0.3/training/"
SAVE_DIR  = Path("ekg_datasets/challenge2021")
DATASETS  = ["georgia", "cpsc_2018", "cpsc_2018_extra", "ningbo"]
N_THREADS = 16   # parallel download threads

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
        print(f"  ERROR listing {url}: {e}", flush=True)
        return []


def download_file(args):
    url, dest = args
    if dest.exists():
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return "ok"
    except Exception as e:
        return f"err:{e}"


def download_dataset(name):
    base = BASE_URL + name + "/"
    save = SAVE_DIR / name
    print(f"\n[{name}] Listing files...", flush=True)

    subdirs = list_dir(base)
    tasks = []
    for sub in subdirs:
        if sub.endswith("/") and sub != "../":
            sub_url = base + sub
            sub_files = list_dir(sub_url)
            for f in sub_files:
                if f.endswith(".mat") or f.endswith(".hea"):
                    tasks.append((sub_url + f, save / sub.rstrip("/") / f))
        elif sub.endswith(".mat") or sub.endswith(".hea"):
            tasks.append((base + sub, save / sub))

    total = len(tasks)
    already = sum(1 for _, d in tasks if d.exists())
    print(f"[{name}] {total} files  ({already} already downloaded, {total-already} to fetch)", flush=True)

    if total == already:
        print(f"[{name}] Already complete - skipping", flush=True)
        return already // 2

    downloaded = 0
    errors     = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        futures = {pool.submit(download_file, t): t for t in tasks}
        done = 0
        for fut in as_completed(futures):
            result = fut.result()
            done += 1
            if result == "ok":
                downloaded += 1
            elif result.startswith("err"):
                errors += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = downloaded / max(elapsed, 1)
                remaining = (total - already - downloaded) / max(rate, 0.1)
                print(f"  [{name}] {done}/{total}  {rate:.0f} new/s  ~{remaining/60:.0f} min left"
                      f"  errors={errors}", flush=True)

    mat_count = len(list(save.rglob("*.mat")))
    elapsed = time.time() - t0
    print(f"[{name}] Done: {mat_count} .mat files  ({downloaded} new, {errors} errors, {elapsed:.0f}s)",
          flush=True)
    return mat_count


if __name__ == "__main__":
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {SAVE_DIR.resolve()}", flush=True)
    print(f"Datasets:  {DATASETS}", flush=True)
    print(f"Threads:   {N_THREADS}", flush=True)

    t_start = time.time()
    for ds in DATASETS:
        download_dataset(ds)

    total_mat = len(list(SAVE_DIR.rglob("*.mat")))
    print(f"\nAll done: {total_mat} total .mat files in {SAVE_DIR}")
    print(f"Total time: {(time.time()-t_start)/60:.0f} min")
