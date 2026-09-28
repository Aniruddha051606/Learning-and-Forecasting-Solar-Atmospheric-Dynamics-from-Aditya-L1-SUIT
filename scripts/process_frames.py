"""Per-frame measurements for the Phase 1 audit, QC and registration study.

    python scripts/process_frames.py [--limit N]

Reads <out>/manifest.parquet. For every full-disk frame: limb fits (distortion-aware and plain
circle), artifact counts, seam step profiles, image statistics (global / disk / off-limb), and for
consecutive NB03 binned frames the measured frame-to-frame image motion (phase correlation) plus a
spike-persistence check. ROI frames get global statistics only; they are an operator-selected sample
and are kept out of every disk-level analysis.

Writes frames_full.parquet, frames_roi.parquet, seam_profiles.parquet and process_meta.json.
"""
import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, median_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import atomic, config, geometry, io, progress, qc  # noqa: E402

CFG = config.load()
MAX_PAIR_DT_S = 300


def phase_motion(prev, cur, box):
    """Image motion of `cur` relative to `prev` (px, +x/+y means features moved to larger x/y), from
    phase correlation of high-passed crops of the same detector box. The box avoids the quadrant seam
    and the limb, so the seam (fixed on the CCD) cannot pull the answer towards zero."""
    from skimage.registration import phase_cross_correlation
    y0, y1, x0, x1 = box
    out = []
    for im in (prev, cur):
        c = im[y0:y1, x0:x1].astype(np.float64)
        c = c - gaussian_filter(c, 15)
        s = 1.4826 * np.median(np.abs(c - np.median(c)))
        out.append(np.clip(c, -6 * s, 6 * s) * np.outer(np.hanning(c.shape[0]), np.hanning(c.shape[1])))
    shift, err, _ = phase_cross_correlation(out[0], out[1], upsample_factor=20, normalization=None)
    # skimage returns the shift that registers cur onto prev: features moved by -shift
    return float(-shift[1]), float(-shift[0]), float(err)


def full_frame(job):
    row, prev_path, check_spikes = job
    L, Q = CFG["limb"], CFG["qc"]
    t0 = time.time()
    out = {"file": row["file"]}
    try:
        im, _ = io.read(row["path"])
        s = io.scale_of(im.shape)
        geo = geometry.limb(im, row["CRPIX1"] - 1, row["CRPIX2"] - 1, row["R_SUN"], rays=L["rays"], window=L["window"],
                            smooth_px=L["smooth_px"], edge_margin_px=L["edge_margin_px"], harmonics=L["harmonics"],
                            clip_sigma=L["clip_sigma"], scale=s)
        if geo is None:
            out["error"] = "limb fit: too few usable rays"
            return out, []
        f, c = geo["limb"], geo["circle"]
        out.update({f"limb_{k}": v for k, v in f.items() if k != "harm"})
        out["limb_harm"] = f["harm"]
        out.update({f"circ_{k}": c[k] for k in ("x0", "y0", "R", "rms", "n_used")})
        e = geo["edges"]
        out["edge_x"] = e["x"].astype(np.float32).tolist()
        out["edge_y"] = e["y"].astype(np.float32).tolist()
        out["edge_usable"] = e["usable"].tolist()
        r = geometry.r_map_model(im.shape, f)
        core = r < Q["core_r"]
        spk, sig = qc.spikes(im, core, Q["spike_k"], Q["spike_rel"], Q["spike_max_area"])
        edge = int(L["edge_margin_px"] * s)
        m = qc.pixel_mask(im, r, row["clip_lo"], row["clip_hi"], spk, Q["seam_px"], Q["seam_halfwidth"], edge)
        bad = (m & (qc.CLIP_LO | qc.CLIP_HI | qc.SPIKE | qc.SEAM)) > 0
        out.update(noise_sigma=float(sig), n_pix=int(im.size),
                   n_clip_lo=int((m & qc.CLIP_LO).astype(bool).sum()), n_clip_hi=int((m & qc.CLIP_HI).astype(bool).sum()),
                   n_spike=int(spk.sum()), n_spike_disk=int((spk & (r < 1)).sum()),
                   n_clip_hi_disk=int(((m & qc.CLIP_HI) > 0)[r < 1].sum()),
                   disk_on_ccd=float((r <= 1).sum() / (np.pi * f["R"] ** 2)),
                   disk_in_edge_margin=float(((m & qc.EDGE) > 0)[r <= 1].mean()),
                   offlimb_neg_frac=float((im[(r > 1.05) & (r < 1.3)] < 0).mean()))
        out.update(qc.stats(im.ravel(), "all"))
        out.update(qc.stats(im[core & ~bad], "disk"))
        out.update(qc.stats(im[(r > 1.05) & (r < 1.3) & ((m & qc.EDGE) == 0)], "offlimb"))
        seams = []
        cpx, _ = qc.seam_boundaries(im.shape, Q["seam_px"])
        for axis, name in ((1, "vertical"), (0, "horizontal")):
            pos, step = qc.seam_profile(im, r < 0.95, cpx, axis, band=int(64 * s), gap=(int(11 * s), int(5 * s)),
                                        width=int(6 * s))
            ctrl_pos, ctrl = qc.seam_profile(im, r < 0.95, cpx + int(150 * s), axis, band=int(64 * s),
                                             gap=(int(11 * s), int(5 * s)), width=int(6 * s))
            out[f"seam_{name}_median"] = float(np.nanmedian(step)) if np.isfinite(step).any() else np.nan
            out[f"seam_{name}_absmax"] = float(np.nanmax(np.abs(step))) if np.isfinite(step).any() else np.nan
            out[f"seam_{name}_control_median"] = float(np.nanmedian(ctrl)) if np.isfinite(ctrl).any() else np.nan
            seams += [{"file": row["file"], "axis": name, "pos": float(p), "step": float(v), "control": float(w)}
                      for p, v, w in zip(pos, step, ctrl)]
        if prev_path is not None:
            prev, _ = io.read(prev_path)
            cy = int(round(f["y0"]))
            box = (max(cy - 288, 0), cy + 288, int(1040 * s), int((1040 + 576) * s))
            out["pc_dx"], out["pc_dy"], out["pc_err"] = phase_motion(prev, im, box)
            if check_spikes and spk.any():
                # a cosmic ray is gone 87 s later; a solar feature is still there
                mp = median_filter(prev, 5)
                resp = prev - mp
                yy, xx = np.nonzero(spk)
                yp = np.clip(np.round(yy - out["pc_dy"]).astype(int), 0, im.shape[0] - 1)
                xp = np.clip(np.round(xx - out["pc_dx"]).astype(int), 0, im.shape[1] - 1)
                out["spike_persist_frac"] = float((resp[yp, xp] > 0.5 * Q["spike_k"] * sig).mean())
        out["seconds"] = round(time.time() - t0, 2)
        return out, seams
    except Exception:
        out["error"] = traceback.format_exc()[-600:]
        return out, []


def roi_frame(row):
    Q = CFG["qc"]
    out = {"file": row["file"]}
    try:
        im, _ = io.read(row["path"])
        spk, sig = qc.spikes(im, np.ones(im.shape, bool), Q["spike_k"], Q["spike_rel"], Q["spike_max_area"])
        out.update(noise_sigma=float(sig), n_spike=int(spk.sum()), n_clip_lo=int((im <= row["clip_lo"]).sum()),
                   n_clip_hi=int((im >= row["clip_hi"]).sum()), n_pix=int(im.size))
        out.update(qc.stats(im[~spk], "all"))
    except Exception:
        out["error"] = traceback.format_exc()[-600:]
    return out


def proc_signature():
    """Hash of the settings that change per-frame results; rows made with other settings are redone."""
    import hashlib
    return hashlib.sha256(json.dumps({"limb": CFG["limb"], "qc": CFG["qc"]}, sort_keys=True).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only the first N frames of each kind (smoke test)")
    ap.add_argument("--full", action="store_true", help="ignore earlier results and process every frame")
    ap.add_argument("--span-of", default=None, help="only frames inside this data set's time span (the rest is kept)")
    ap.add_argument("--skip-roi", action="store_true", help="do not process ROI frames (the NB03 pipeline never uses them)")
    a = ap.parse_args()
    out = config.out_dir(CFG)
    sig = proc_signature()
    man = pd.read_parquet(out / "manifest.parquet")
    man = man[man.read_error.isna()].sort_values("t")
    if a.span_of:
        lo, hi = config.dataset_span(config.load_dataset(a.span_of))
        man = man[(man.t >= lo) & (man.t <= hi)]
    full = man[man.frame.isin(["full", "full_binned"])].reset_index(drop=True)
    roi = man[man.frame == "roi"].reset_index(drop=True)
    if a.limit:
        full, roi = full.head(a.limit), roi.head(a.limit)

    nb = full[full.frame == "full_binned"].sort_values("t")
    prev_of = {}
    for k, (i, r) in enumerate(nb.iterrows()):
        if k and (r.t - nb.iloc[k - 1].t).total_seconds() <= MAX_PAIR_DT_S:
            prev_of[r.file] = (nb.iloc[k - 1].path, k % 10 == 0)
    full["prev_file"] = full.file.map(lambda f: Path(prev_of[f][0]).name if f in prev_of else None)

    # Incremental: reuse rows computed from the same file content, settings and previous-frame pairing.
    old_full = old_roi = old_seams = None
    if not a.full and not a.limit and (out / "frames_full.parquet").exists():
        old_full = pd.read_parquet(out / "frames_full.parquet")
        old_roi = pd.read_parquet(out / "frames_roi.parquet")
        old_seams = pd.read_parquet(out / "seam_profiles.parquet")
        sha = man.set_index("file").sha256
        for df in (old_full, old_roi):
            if "sha256" not in df:  # results from before incremental runs: made with the current settings
                df["sha256"], df["proc_sig"] = df.file.map(sha), sig
            if "prev_file" not in df:
                df["prev_file"] = None
        cur = full.set_index("file")
        ok = (old_full.proc_sig.eq(sig) & old_full.file.isin(cur.index)
              & old_full.sha256.eq(old_full.file.map(sha))
              & (old_full.pc_dx.isna() == old_full.file.map(cur.prev_file).isna()))
        old_full = old_full[ok]
        old_roi = old_roi[old_roi.proc_sig.eq(sig) & old_roi.sha256.eq(old_roi.file.map(sha))]
        old_seams = old_seams[old_seams.file.isin(old_full.file)]
    done_full = set() if old_full is None else set(old_full.file)
    done_roi = set() if old_roi is None else set(old_roi.file)

    jobs = []
    for _, r in full[~full.file.isin(done_full)].iterrows():
        p, chk = prev_of.get(r.file, (None, False))
        jobs.append((r.to_dict(), p, chk))
    roi_todo = roi[~roi.file.isin(done_roi)] if not a.skip_roi else roi.iloc[0:0]
    print(f"full-disk: {len(jobs)} to process, {len(done_full)} reused; ROI: {len(roi_todo)} to process, "
          f"{len(done_roi)} reused", flush=True)

    t0 = time.time()
    keep = ["file", "t", "obsid", "OBS_MODE", "FTR_NAME", "frame", "NAXIS1", "CMD_EXPT", "MEAS_EXP", "CRPIX1", "CRPIX2",
            "R_SUN", "CROTA2", "RSUN_OBS", "DSUN_OBS", "clip_lo", "clip_hi", "sha256"]

    def save(res, roi_res):
        """Earlier rows + everything finished so far, written atomically. Called every few minutes and at the
        end, so a crash (or a killed run) loses at most a few minutes: the next run reuses the saved rows."""
        new = pd.DataFrame([r for r, _ in res]) if res else pd.DataFrame(columns=["file"])
        new_seams = pd.DataFrame([x for _, ss in res for x in ss])
        new = full[keep + ["prev_file"]].merge(new, on="file", how="inner").assign(proc_sig=sig)
        new_roi = roi_todo[keep + ["ROI_ID", "X1", "Y1"]].merge(pd.DataFrame(roi_res or [{"file": None}]), on="file",
                                                                 how="inner").assign(proc_sig=sig)
        fr_ = pd.concat([d for d in (old_full, new) if d is not None and len(d)], ignore_index=True).sort_values("t")
        ro_ = pd.concat([d for d in (old_roi, new_roi) if d is not None and len(d)], ignore_index=True).sort_values("t")
        se_ = pd.concat([d for d in (old_seams, new_seams) if d is not None and len(d)], ignore_index=True)
        atomic.to_parquet(fr_, out / "frames_full.parquet")
        atomic.to_parquet(ro_, out / "frames_roi.parquet")
        atomic.to_parquet(se_, out / "seam_profiles.parquet")
        return fr_, ro_

    res, roi_res, last_save = [], [], time.time()
    roi_jobs = [r.to_dict() for _, r in roi_todo.iterrows()]
    with ProcessPoolExecutor(CFG["run"]["workers"]) as ex:
        for k, result in enumerate(ex.map(full_frame, jobs, chunksize=4)):
            res.append(result)
            progress.report("process_frames: full-disk", item=jobs[k][0].get("file"), i=k, n=len(jobs),
                            path=jobs[k][0].get("path"))
            if time.time() - last_save > 300:
                save(res, roi_res)
                last_save = time.time()
        for k, result in enumerate(ex.map(roi_frame, roi_jobs, chunksize=32)):
            roi_res.append(result)
            progress.report("process_frames: ROI", item=roi_jobs[k].get("file"), i=k, n=len(roi_jobs),
                            path=roi_jobs[k].get("path"))
            if time.time() - last_save > 300:
                save(res, roi_res)
                last_save = time.time()
    frames, roi_df = save(res, roi_res)
    meta = {"full_frames": len(frames), "roi_frames": len(roi_df), "processed_now": [len(jobs), len(roi_todo)],
            "full_errors": int(frames.get("error", pd.Series(dtype=object)).notna().sum()),
            "roi_errors": int(roi_df.get("error", pd.Series(dtype=object)).notna().sum()),
            "proc_sig": sig, "seconds": round(time.time() - t0, 1), "manifest_rows": len(man), **CFG["_meta"]}
    atomic.write_json(out / "process_meta.json", meta)
    print(json.dumps(meta, indent=1))
    if meta["full_errors"]:
        print(frames.loc[frames.error.notna(), ["file", "error"]].head(5).to_string())


if __name__ == "__main__":
    main()
