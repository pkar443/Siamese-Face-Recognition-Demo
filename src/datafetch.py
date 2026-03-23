import argparse
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pandas as pd

from utils import DEFAULT_EVAL_DIR, DEFAULT_TRAIN_DIR, resolve_path

DEFAULT_DEV_URL = "http://www.cs.columbia.edu/CAVE/databases/pubfig/download/dev_urls.txt"
DEFAULT_EVAL_URL = "http://www.cs.columbia.edu/CAVE/databases/pubfig/download/eval_urls.txt"
DEFAULT_TIMEOUT = 3.0
DEFAULT_WORKERS = min(16, max(4, (os.cpu_count() or 4) * 2))
DEFAULT_PROGRESS_EVERY = 100
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SiameseFaceDemo/1.0)",
    "Connection": "close",
}


def _read_pubfig_index(url, max_rows=None):
    data = pd.read_csv(
        url,
        sep="\t",
        skiprows=2,
        header=None,
        names=["name", "image_num", "url", "rect", "md5"],
    )
    data = data.dropna(subset=["name", "url", "rect"])
    if max_rows:
        data = data.head(max_rows)
    return data


def _output_path_for_row(dirname, row_index, person_name):
    return dirname / person_name / f"{int(row_index):06d}.jpg"


def _download_and_process(task, img_shape, timeout):
    request = Request(task["url"], headers=REQUEST_HEADERS)
    try:
        with urlopen(request, timeout=timeout) as response:
            image_bytes = response.read()
    except Exception:
        return False, "download"

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return False, "decode"

    try:
        x1, y1, x2, y2 = map(int, str(task["rect"]).split(","))
    except Exception:
        return False, "rect"

    height, width = image.shape[:2]
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return False, "crop"

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return False, "crop"

    cropped = cv2.resize(cropped, img_shape, interpolation=cv2.INTER_AREA)
    task["output_path"].parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(task["output_path"]), cropped):
        return False, "write"
    return True, None


def _path_is_on_mounted_windows_drive(path):
    parts = resolve_path(path).parts
    return len(parts) > 1 and parts[1] == "mnt"


def _remove_empty_person_dirs(dirname):
    removed = 0
    if not dirname.exists():
        return removed
    for person_dir in dirname.iterdir():
        if person_dir.is_dir() and not any(person_dir.iterdir()):
            person_dir.rmdir()
            removed += 1
    return removed


def _count_people_with_images(dirname):
    people = 0
    if not dirname.exists():
        return people
    for person_dir in dirname.iterdir():
        if not person_dir.is_dir():
            continue
        if any(path.is_file() for path in person_dir.iterdir()):
            people += 1
    return people


def get_data(
    url,
    dirname,
    img_shape=(100, 100),
    max_rows=None,
    timeout=DEFAULT_TIMEOUT,
    workers=DEFAULT_WORKERS,
    progress_every=DEFAULT_PROGRESS_EVERY,
    log_callback=print,
):
    data = _read_pubfig_index(url, max_rows=max_rows)
    dirname = resolve_path(dirname)
    dirname.mkdir(parents=True, exist_ok=True)

    total_rows = int(data.shape[0])
    total_people = int(data["name"].nunique())
    workers = max(1, int(workers))
    timeout = float(timeout)
    progress_every = max(1, int(progress_every))

    pending_tasks = []
    skipped_existing = 0
    for row_index, row in data.iterrows():
        output_path = _output_path_for_row(dirname, row_index, row["name"])
        if output_path.exists():
            skipped_existing += 1
            continue
        pending_tasks.append(
            {
                "url": row["url"],
                "rect": row["rect"],
                "output_path": output_path,
            }
        )

    log_callback(
        f"{dirname.name}: {total_rows} rows across {total_people} people, "
        f"{len(pending_tasks)} pending, {skipped_existing} already present, "
        f"workers={workers}, timeout={timeout:.1f}s"
    )

    if not pending_tasks:
        log_callback(f"{dirname.name}: dataset already present, nothing to download.")
        return {
            "rows": total_rows,
            "people": total_people,
            "saved": 0,
            "failed": 0,
            "skipped_existing": skipped_existing,
            "failure_reasons": {},
        }

    saved = 0
    failed = 0
    failure_reasons = Counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_download_and_process, task, img_shape, timeout) for task in pending_tasks]
        total_pending = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            ok, reason = future.result()
            if ok:
                saved += 1
            else:
                failed += 1
                failure_reasons[reason] += 1

            if completed % progress_every == 0 or completed == total_pending:
                log_callback(
                    f"{dirname.name}: completed {completed}/{total_pending}, "
                    f"saved={saved}, failed={failed}, skipped_existing={skipped_existing}"
                )

    if failure_reasons:
        top_reasons = ", ".join(f"{name}={count}" for name, count in failure_reasons.most_common(4))
        log_callback(f"{dirname.name}: failure summary -> {top_reasons}")

    removed_empty = _remove_empty_person_dirs(dirname)
    if removed_empty:
        log_callback(f"{dirname.name}: removed {removed_empty} empty identity folders")
    usable_people = _count_people_with_images(dirname)
    log_callback(f"{dirname.name}: usable identities with at least one saved image -> {usable_people}")

    return {
        "rows": total_rows,
        "people": total_people,
        "saved": saved,
        "failed": failed,
        "skipped_existing": skipped_existing,
        "failure_reasons": dict(failure_reasons),
        "usable_people": usable_people,
        "removed_empty": removed_empty,
    }


def run_fetch(
    train_dir=DEFAULT_TRAIN_DIR,
    eval_dir=DEFAULT_EVAL_DIR,
    max_rows=None,
    train_url=DEFAULT_EVAL_URL,
    eval_url=DEFAULT_DEV_URL,
    timeout=DEFAULT_TIMEOUT,
    workers=DEFAULT_WORKERS,
    progress_every=DEFAULT_PROGRESS_EVERY,
    log_callback=print,
):
    log_lock = threading.Lock()
    results = {}

    def thread_safe_log(message):
        with log_lock:
            log_callback(message)

    def fetch_split(split_name, url, output_dir):
        results[split_name] = get_data(
            url=url,
            dirname=output_dir,
            img_shape=(100, 100),
            max_rows=max_rows,
            timeout=timeout,
            workers=workers,
            progress_every=progress_every,
            log_callback=thread_safe_log,
        )

    if _path_is_on_mounted_windows_drive(train_dir) or _path_is_on_mounted_windows_drive(eval_dir):
        thread_safe_log(
            "Note: dataset is being written under /mnt. Many small image writes are slower there than in native Linux storage."
        )

    eval_thread = threading.Thread(target=fetch_split, args=("eval", eval_url, eval_dir), daemon=True)
    train_thread = threading.Thread(target=fetch_split, args=("train", train_url, train_dir), daemon=True)

    eval_thread.start()
    train_thread.start()
    train_thread.join()
    eval_thread.join()

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Download PubFig training and eval images.")
    parser.add_argument("--train-url", default=DEFAULT_EVAL_URL, help="URL used to build the train folder")
    parser.add_argument("--eval-url", default=DEFAULT_DEV_URL, help="URL used to build the eval folder")
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN_DIR), help="Output train directory")
    parser.add_argument("--eval-dir", default=str(DEFAULT_EVAL_DIR), help="Output eval directory")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests before a full download",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Per-image download timeout in seconds",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel download workers per split",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help="How often to print progress updates",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_fetch(
        train_dir=args.train_dir,
        eval_dir=args.eval_dir,
        max_rows=args.max_rows,
        train_url=args.train_url,
        eval_url=args.eval_url,
        timeout=args.timeout,
        workers=args.workers,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
