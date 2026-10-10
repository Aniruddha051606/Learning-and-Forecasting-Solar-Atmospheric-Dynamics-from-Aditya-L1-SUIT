"""Phase 1 exploratory analysis: per-filter audit tables and diagnostic plots.

    python scripts/eda.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from suitdyn import config, geometry, io  # noqa: E402
from suitdyn.filters import FILTERS, ORDER  # noqa: E402

CFG = config.load()
OUT = config.out_dir(CFG)
EDA = config.out_dir(CFG, "eda")
PROG_COLORS = {139: "C0", 147: "C1", 151: "C2"}


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(EDA / name, dpi=90)
    plt.close(fig)


def cadence_table(man, frames, roi):
    rows = []
    qc_ok = frames.set_index("file").get("qc_usable")
    for (fr, f), g in man[man.read_error.isna()].groupby(["frame", "FTR_NAME"]):
        t = g.t.sort_values()
        dt = t.diff().dt.total_seconds().dropna()
        usable = int(qc_ok.reindex(g.file).fillna(False).sum()) if qc_ok is not None and fr != "roi" else np.nan
        rows.append({"frame": fr, "filter": f, "wavelength_nm": FILTERS[f][0], "frames": len(g),
                     "usable": usable, "invalid": len(g) - usable if usable == usable else np.nan,
                     "cadence_median_s": dt.median(), "cadence_mean_s": dt.mean(), "cadence_std_s": dt.std(),
                     "cadence_p95_s": dt.quantile(.95), "largest_gap_h": dt.max() / 3600,
                     "duration_h": (t.max() - t.min()).total_seconds() / 3600, "gigabytes": g.bytes.sum() / 1e9})
    tab = pd.DataFrame(rows)
    img = frames.groupby(["frame", "FTR_NAME"])[[c for c in frames.columns if c.startswith(("disk_", "all_", "offlimb_"))
                                                  and c.split("_")[-1] in ("mean", "median", "std", "min", "max", "p1",
                                                                          "p5", "p95", "p99")]].median()
    img.index = img.index.set_names(["frame", "filter"])
    roi_img = roi.groupby(["frame", "FTR_NAME"])[[c for c in roi.columns if c.startswith("all_")]].median()
    roi_img.index = roi_img.index.set_names(["frame", "filter"])
    img = pd.concat([img, roi_img])
    return tab.merge(img.reset_index(), on=["frame", "filter"], how="left")


def plot_timeline(man):
    m = man[man.read_error.isna()]
    fig, ax = plt.subplots(figsize=(15, 6))
    ylab = []
    for k, (fr, f) in enumerate([(fr, f) for fr in ("full_binned", "full", "roi") for f in ORDER
                                 if ((m.frame == fr) & (m.FTR_NAME == f)).any()]):
        g = m[(m.frame == fr) & (m.FTR_NAME == f)]
        for prog, gg in g.groupby("OBS_MODE"):
            ax.plot(gg.t, np.full(len(gg), k), "|", ms=10, color=PROG_COLORS.get(prog, "k"))
        ylab.append(f"{fr} {f}")
    ax.set_yticks(range(len(ylab)))
    ax.set_yticklabels(ylab, fontsize=7)
    for p, c in PROG_COLORS.items():
        ax.plot([], [], "|", color=c, label=f"OBS_MODE {p}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Observation timeline by frame type and filter (colour = observing program)")
    savefig(fig, "01_timeline.png")


def plot_cadence(man):
    m = man[man.read_error.isna()]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    nb = m[m.frame == "full_binned"].sort_values("t")
    dt = nb.t.diff().dt.total_seconds()
    ax[0].hist(dt.dropna().clip(upper=600), bins=120)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("NB03 binned Δt (s, clipped at 600)")
    ax[1].semilogy(nb.t, dt, ".", ms=2)
    ax[1].set_ylabel("Δt to previous NB03 frame (s)")
    ax[1].set_title("NB03 cadence and gaps")
    full = m[m.frame == "full"].sort_values("t")
    for f in ORDER:
        g = full[full.FTR_NAME == f]
        ax[2].plot(g.t, np.full(len(g), ORDER.index(f)), "o", ms=3)
    ax[2].set_yticks(range(len(ORDER)))
    ax[2].set_yticklabels(ORDER, fontsize=7)
    ax[2].set_title("Unbinned full-disk frames (multi-filter bursts)")
    for a in ax[1:]:
        a.tick_params(axis="x", labelrotation=30, labelsize=7)
    savefig(fig, "02_cadence_gaps.png")


def plot_intensity(frames):
    nb = frames[(frames.frame == "full_binned") & frames.limb_R.notna()].sort_values("t")
    fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for prog, g in nb.groupby("OBS_MODE"):
        c = PROG_COLORS.get(prog, "k")
        ax[0].plot(g.t, g.disk_median, ".", ms=2, color=c, label=f"program {prog}")
        ax[1].plot(g.t, g.disk_p99, ".", ms=2, color=c)
        ax[2].plot(g.t, g.offlimb_median, ".", ms=2, color=c)
    ax[0].set_ylabel("disk median (r<0.9, counts)")
    ax[0].legend(fontsize=8)
    ax[1].set_ylabel("disk p99")
    ax[2].set_ylabel("off-limb median (1.05<r<1.3)")
    ax[0].set_title("NB03 binned intensity statistics")
    savefig(fig, "03_intensity_nb03.png")

    full = frames[(frames.frame == "full") & frames.limb_R.notna()]
    fig, ax = plt.subplots(figsize=(15, 5))
    for f in ORDER:
        g = full[full.FTR_NAME == f].sort_values("t")
        if len(g):
            ax.plot(g.t, g.disk_median / g.disk_median.median(), "o-", ms=3, lw=.6, label=f)
    ax.set_ylabel("disk median / filter median")
    ax.legend(ncol=6, fontsize=7)
    ax.set_title("Unbinned full-disk frames: relative disk brightness per filter")
    savefig(fig, "04_intensity_filters.png")


def plot_pointing(frames, reg):
    nb = frames[(frames.frame == "full_binned") & frames.limb_R.notna()].sort_values("t")
    fig, ax = plt.subplots(4, 1, figsize=(15, 13), sharex=True)
    for k, (col, lab) in enumerate((("x0", "x"), ("y0", "y"))):
        ax[k].plot(nb.t, nb.CRPIX1 - 1 if lab == "x" else nb.CRPIX2 - 1, ".", ms=2, label="header CRPIX")
        ax[k].plot(nb.t, nb[f"circ_{col}"], ".", ms=2, label="limb: circle")
        ax[k].plot(nb.t, nb[f"limb_{col}"], ".", ms=2, label="limb: circle + harmonics")
        if reg is not None:
            r = reg.set_index("file").reindex(nb.file)
            ax[k].plot(nb.t, r[f"reg_{col}"].values, "k-", lw=.8, label="adopted (registration)")
        ax[k].set_ylabel(f"disk centre {lab} (px)")
        ax[k].legend(fontsize=7, ncol=4)
    ax[2].plot(nb.t, nb.R_SUN, ".", ms=2, label="header R_SUN")
    ax[2].plot(nb.t, nb.circ_R, ".", ms=2, label="circle")
    ax[2].plot(nb.t, nb.limb_R, ".", ms=2, label="harmonic mean R")
    ax[2].set_ylabel("disk radius (px)")
    ax[2].legend(fontsize=7)
    mot = np.hypot(nb.pc_dx, nb.pc_dy)
    ax[3].semilogy(nb.t, mot.clip(lower=1e-2), ".", ms=2)
    ax[3].set_ylabel("|frame-to-frame image motion| (px)\nphase correlation")
    ax[0].set_title("NB03 binned: pointing and disk geometry")
    savefig(fig, "05_pointing.png")


def plot_geometry(frames):
    f = frames[frames.limb_R.notna()]
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    for fr, g in f.groupby("frame"):
        ax[0, 0].plot(g.t, g.disk_on_ccd, ".", ms=3, label=fr)
        ax[0, 1].plot(g.t, g.limb_rms, ".", ms=3, label=f"{fr} harmonic")
        ax[0, 1].plot(g.t, g.circ_rms, "x", ms=3, label=f"{fr} circle")
        ax[1, 0].plot(g.t, g.limb_angular_coverage, ".", ms=3, label=fr)
    ax[0, 0].set_ylabel("fraction of disk on the detector")
    ax[0, 1].set_ylabel("limb-fit residual rms (px)")
    ax[1, 0].set_ylabel("fraction of limb rays usable")
    nb = f[f.frame == "full_binned"]
    H = np.array(nb.limb_harm.tolist())
    for j, m in enumerate(range(2, 2 + H.shape[1] // 2)):
        ax[1, 1].plot(nb.t, np.hypot(H[:, 2 * j], H[:, 2 * j + 1]), ".", ms=2, label=f"m={m}")
    ax[1, 1].set_ylabel("NB03 limb distortion amplitude (px)")
    for a in ax.ravel():
        a.legend(fontsize=7)
        a.tick_params(axis="x", labelrotation=30, labelsize=7)
    savefig(fig, "06_disk_geometry.png")


def plot_artifacts(frames, seams):
    f = frames[frames.limb_R.notna()]
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))
    for fr, g in f.groupby("frame"):
        ax[0, 0].semilogy(g.t, g.n_spike / g.n_pix * 1e6, ".", ms=3, label=fr)
        ax[0, 1].plot(g.t, g.n_clip_hi, ".", ms=3, label=f"{fr} clipped high")
        ax[1, 0].plot(g.t, g.offlimb_neg_frac, ".", ms=3, label=fr)
    ax[0, 0].set_ylabel("spikes per million pixels")
    ax[0, 1].set_ylabel("pixels at the encoding ceiling")
    ax[1, 0].set_ylabel("off-limb fraction < 0")
    nbfiles = set(f.loc[f.frame == "full_binned", "file"])
    s = seams[seams.file.isin(nbfiles) & (seams.axis == "vertical")]
    gb = s.groupby("pos")
    med, q10, q90 = gb.step.median(), gb.step.quantile(.1), gb.step.quantile(.9)
    ax[1, 1].plot(med.index, med * 100, "r-", label="seam x=1023|1024 (median)")
    ax[1, 1].fill_between(med.index, q10 * 100, q90 * 100, color="r", alpha=.2, label="10-90 %")
    ax[1, 1].plot(med.index, gb.control.median() * 100, "k--", label="control columns (+150 px)")
    ax[1, 1].set_xlabel("row (binned px)")
    ax[1, 1].set_ylabel("step across boundary (%)")
    ax[1, 1].set_title("NB03 vertical quadrant seam")
    for a in ax.ravel():
        a.legend(fontsize=7)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.tick_params(axis="x", labelrotation=30, labelsize=7)
    savefig(fig, "07_artifacts.png")


def stretch(im, lo=1, hi=99.7):
    v = im[np.isfinite(im)]
    a, b = np.percentile(v, [lo, hi])
    return np.clip((im - a) / (b - a), 0, 1)


def plot_samples(frames):
    full = frames[(frames.frame == "full") & frames.limb_R.notna()].sort_values("t")
    if full.empty:
        return
    burst_t = full.t.iloc[len(full) // 2]
    burst = full[(full.t - burst_t).abs() < pd.Timedelta("6min")].drop_duplicates("FTR_NAME")
    man = pd.read_parquet(OUT / "manifest.parquet", columns=["file", "path"]).set_index("file")
    fig, ax = plt.subplots(3, 4, figsize=(16, 12))
    profiles = {}
    for a, f in zip(ax.ravel(), ORDER):
        row = burst[burst.FTR_NAME == f]
        a.set_axis_off()
        if row.empty:
            continue
        row = row.iloc[0]
        im, _ = io.read(man.loc[row.file, "path"])
        a.imshow(stretch(im[::4, ::4]), origin="lower", cmap="inferno")
        a.set_title(f"{f} {FILTERS[f][0]} nm {FILTERS[f][2]}\n{row.t:%m-%d %H:%M:%S} exp {row.CMD_EXPT} ms", fontsize=8)
        fit = {"x0": row.limb_x0, "y0": row.limb_y0, "R": row.limb_R, "harm": list(row.limb_harm)}
        r = geometry.r_map_model(im.shape, fit)
        bins = np.linspace(0, 1.3, 131)
        idx = np.digitize(r.ravel(), bins)
        med = pd.Series(im.ravel()).groupby(idx).median()
        profiles[f] = (bins[np.clip(med.index - 1, 0, len(bins) - 1)], med.values / np.nanmedian(im[r < .3]))
    ax.ravel()[-1].set_axis_on()
    for f, (x, y) in profiles.items():
        ax.ravel()[-1].plot(x, y, lw=.8, label=f)
    ax.ravel()[-1].set_xlim(0, 1.3)
    ax.ravel()[-1].set_ylim(-0.3, 1.5)
    ax.ravel()[-1].axvline(1, color="k", lw=.5)
    ax.ravel()[-1].set_xlabel("r / R (fitted limb)")
    ax.ravel()[-1].set_title("centre-to-limb profile / disk-centre level", fontsize=8)
    ax.ravel()[-1].legend(fontsize=6, ncol=2)
    fig.suptitle(f"One multi-filter burst, around {burst_t:%Y-%m-%d %H:%M} UT (4096² frames shown at 1/4)")
    savefig(fig, "08_filters_burst.png")


def main():
    man = pd.read_parquet(OUT / "manifest.parquet")
    frames = pd.read_parquet(OUT / "frames_full.parquet")
    roi = pd.read_parquet(OUT / "frames_roi.parquet")
    seams = pd.read_parquet(OUT / "seam_profiles.parquet")
    regp = OUT / "registration.parquet"
    reg = pd.read_parquet(regp) if regp.exists() else None
    if reg is not None:
        frames = frames.merge(reg[["file", "qc_usable", "qc_reasons"]], on="file", how="left")
    tab = cadence_table(man, frames, roi)
    tab.to_csv(OUT / "audit_filters.csv", index=False)
    plot_timeline(man)
    plot_cadence(man)
    plot_intensity(frames)
    plot_pointing(frames, reg)
    plot_geometry(frames)
    plot_artifacts(frames, seams)
    plot_samples(frames)
    pd.set_option("display.width", 250)
    print(tab[["frame", "filter", "frames", "usable", "invalid", "cadence_median_s", "cadence_p95_s", "largest_gap_h",
               "duration_h", "gigabytes"]].round(2).to_string())
    print("plots:", sorted(p.name for p in EDA.glob("*.png")))


if __name__ == "__main__":
    main()
