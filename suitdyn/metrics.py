"""Forecast metrics on valid pixels only."""
import numpy as np
from scipy.ndimage import gaussian_filter, sobel


def _m(f, t, mask):
    m = np.isfinite(f) & np.isfinite(t)
    if mask is not None:
        m &= mask
    return m


def mae(f, t, mask=None):
    m = _m(f, t, mask)
    return float(np.mean(np.abs(f[m] - t[m])))


def rmse(f, t, mask=None):
    m = _m(f, t, mask)
    return float(np.sqrt(np.mean((f[m] - t[m]) ** 2)))


def psnr(f, t, mask=None):
    m = _m(f, t, mask)
    rng = np.percentile(t[m], 99.9) - np.percentile(t[m], 0.1)
    return float(20 * np.log10(rng / rmse(f, t, m)))


def ssim(f, t, mask=None, sigma=1.5):
    """Mean SSIM over valid pixels (Gaussian-window SSIM, Wang et al. 2004), data range from the truth."""
    m = _m(f, t, mask)
    rng = np.percentile(t[m], 99.9) - np.percentile(t[m], 0.1)
    c1, c2 = (0.01 * rng) ** 2, (0.03 * rng) ** 2
    ff, tt = np.where(m, f, 0.0), np.where(m, t, 0.0)
    w = gaussian_filter(m.astype(float), sigma)
    w = np.where(w > 1e-6, w, np.nan)
    mu_f, mu_t = gaussian_filter(ff, sigma) / w, gaussian_filter(tt, sigma) / w
    s_ff = gaussian_filter(ff * ff, sigma) / w - mu_f ** 2
    s_tt = gaussian_filter(tt * tt, sigma) / w - mu_t ** 2
    s_ft = gaussian_filter(ff * tt, sigma) / w - mu_f * mu_t
    s = ((2 * mu_f * mu_t + c1) * (2 * s_ft + c2)) / ((mu_f ** 2 + mu_t ** 2 + c1) * (s_ff + s_tt + c2))
    inner = m & (gaussian_filter(m.astype(float), sigma) > 0.99)
    return float(np.nanmean(s[inner]))


def gradient_correlation(f, t, mask=None):
    """Correlation of gradient magnitudes: are edges (network, plage boundaries) where they should be?"""
    m = _m(f, t, mask)
    g = [np.hypot(sobel(np.where(m, a, 0.0), 0), sobel(np.where(m, a, 0.0), 1)) for a in (f, t)]
    inner = m & (gaussian_filter(m.astype(float), 1) > 0.99)
    return float(np.corrcoef(g[0][inner], g[1][inner])[0, 1])


def bright_region(img, mask, level):
    """Bright (plage-like) pixels: above `level` times the median of the valid disk."""
    return (img > level * np.nanmedian(img[mask])) & mask


def region_scores(f, t, mask=None, level=1.3):
    """Morphology of bright regions: IoU of the bright-pixel masks, relative error of their area and of their
    integrated excess brightness.
    """
    m = _m(f, t, mask)
    bf, bt = bright_region(f, m, level), bright_region(t, m, level)
    inter, union = (bf & bt).sum(), (bf | bt).sum()
    med_f, med_t = np.median(f[m]), np.median(t[m])
    ex_f = float(np.sum(f[bf] - med_f))
    ex_t = float(np.sum(t[bt] - med_t))
    return {"bright_iou": float(inter / union) if union else np.nan,
            "bright_area_rel_err": float((bf.sum() - bt.sum()) / bt.sum()) if bt.sum() else np.nan,
            "bright_excess_rel_err": float((ex_f - ex_t) / ex_t) if ex_t else np.nan}


def all_metrics(f, t, mask=None):
    out = {"mae": mae(f, t, mask), "rmse": rmse(f, t, mask), "psnr": psnr(f, t, mask), "ssim": ssim(f, t, mask),
           "grad_corr": gradient_correlation(f, t, mask)}
    out.update(region_scores(f, t, mask))
    return out
