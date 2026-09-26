import numpy as np
from astropy.wcs import WCS

from suitdyn import register


def test_transform_matches_astropy_wcs():
    """Our transform must agree with astropy's reading of the same FITS WCS (HPLN-TAN, CROTA2), with the
    fitted centre and a radius-derived plate scale substituted, to well under a pixel."""
    x0, y0, R_fit, crota, rsun = 1281.4, 597.2, 699.3, -1.93, 972.45
    grid, r_ref = 1536, 690.0
    cdelt = rsun / R_fit
    w = WCS(naxis=2)
    w.wcs.ctype = ["HPLN-TAN", "HPLT-TAN"]
    w.wcs.cunit = ["arcsec", "arcsec"]
    w.wcs.crpix = [x0 + 1, y0 + 1]  # FITS is 1-based
    w.wcs.cdelt = [cdelt, cdelt]
    w.wcs.crval = [0, 0]
    w.wcs.crota = [0, crota]
    A, b = register.transform(x0, y0, R_fit, crota, grid, r_ref)
    c = (grid - 1) / 2
    for u, v in [(c, c), (c + 600, c), (c, c - 650), (c - 400, c + 420)]:
        hp = np.array([u - c, v - c]) * rsun / r_ref
        # astropy: world → pixel (0-based)
        # astropy normalises celestial world coordinates to degrees
        px = np.array(w.wcs_world2pix([[hp[0] / 3600, hp[1] / 3600]], 0)[0])
        ours = A @ np.array([u, v]) + b
        assert np.allclose(ours, px, atol=0.05), (u, v, ours, px)


def test_register_centres_disk_and_normalises_radius():
    """A synthetic disk anywhere on the CCD lands centred on the grid with radius r_ref."""
    from tests.test_geometry import synthetic_disk
    from suitdyn import geometry
    img = synthetic_disk(harm=())
    A, b = register.transform(1280.3, 598.7, 697.0, 0.0, 1536, 690.0)
    out = register.apply(img, A, b, 1536)
    fit = geometry.limb(out, 767.5, 767.5, 690.0, harmonics=4)["limb"]  # NaN = no source pixel
    assert abs(fit["x0"] - 767.5) < 0.3 and abs(fit["y0"] - 767.5) < 0.3 and abs(fit["R"] - 690.0) < 0.5
