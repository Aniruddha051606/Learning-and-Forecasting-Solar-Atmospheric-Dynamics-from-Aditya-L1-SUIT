"""One-way incremental mirror of the SUIT archive of record (network share) to the local working copy.

    python scripts/sync_archive.py            # dry run: lists what would be copied, writes nothing
    python scripts/sync_archive.py --copy     # copy new/changed files, verify each by SHA-256
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config  # noqa: E402

CFG = config.load()


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def scan(root):
    out = {}
    for dp, _, fn in os.walk(root):
        for f in fn:
            if f.endswith(".fits"):
                p = os.path.join(dp, f)
                st = os.stat(p)
                out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true", help="actually copy (default: dry run)")
    ap.add_argument("--settle-s", type=float, default=120, help="skip files modified more recently than this")
    ap.add_argument("--limit", type=int, default=0, help="copy at most N files this run")
    a = ap.parse_args()
    src_root = CFG["paths"].get("archive_root")
    dst_root = CFG["paths"]["raw_root"]
    if not src_root:
        sys.exit("configs/phase1.toml [paths] archive_root is not set")
    t0 = time.time()
    src, dst = scan(src_root), scan(dst_root)
    now = time.time()
    todo, settling = [], 0
    for rel, (size, mtime) in sorted(src.items()):
        if now - mtime < a.settle_s:
            settling += 1
            continue
        if rel not in dst or dst[rel][0] != size:
            todo.append((rel, size))
    if a.limit:
        todo = todo[:a.limit]
    partial = sum(1 for dp, _, fn in os.walk(src_root) for f in fn if f.endswith(".part"))
    summary = {"archive_root": src_root, "raw_root": dst_root, "archive_files": len(src), "local_files": len(dst),
               "to_copy": len(todo), "to_copy_gb": round(sum(s for _, s in todo) / 1e9, 3),
               "skipped_still_settling": settling, "partial_downloads_on_share": partial,
               "only_local": len(set(dst) - set(src)), "mode": "copy" if a.copy else "dry-run"}
    by_obsid = {}
    for rel, s in todo:
        k = Path(rel).parent.name
        by_obsid[k] = by_obsid.get(k, 0) + 1
    summary["to_copy_by_obsid"] = by_obsid
    copied, bad, nbytes = 0, [], 0
    if a.copy:
        for rel, size in todo:
            s, d = os.path.join(src_root, rel), os.path.join(dst_root, rel)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            tmp = d + ".syncpart"
            shutil.copyfile(s, tmp)
            if sha256(tmp) != sha256(s):
                os.remove(tmp)
                bad.append(rel)
                continue
            os.replace(tmp, d)
            copied += 1
            nbytes += size
            if copied % 100 == 0:
                el = time.time() - t0
                print(f"{copied}/{len(todo)} copied, {nbytes / 1e6 / el:.1f} MB/s", flush=True)
    summary.update(copied=copied, checksum_mismatches=bad, seconds=round(time.time() - t0, 1),
                   rate_mb_s=round(nbytes / 1e6 / max(time.time() - t0, 1e-9), 2) if copied else None, **CFG["_meta"])
    out = config.ROOT / "outputs" / "sync"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sync_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **summary}, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("config_path", "git")}, indent=1, default=str))


if __name__ == "__main__":
    main()
