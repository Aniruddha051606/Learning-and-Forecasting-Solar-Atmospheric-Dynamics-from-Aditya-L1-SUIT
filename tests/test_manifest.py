import numpy as np
from astropy.io import fits

from suitdyn import manifest


def _write(path, t, flt="NB03"):
    h = fits.Header()
    h["DATE-OBS"] = t
    h["FTR_NAME"] = flt
    h["ROI_FF"] = "Full Frame"
    h["BIN_EN"] = "enable"
    h["BZERO"], h["BSCALE"] = 30000, 1
    fits.PrimaryHDU(np.zeros((4, 4), np.int16), header=h).writeto(path)


def test_incremental_reuse_keeps_every_column(tmp_path):
    """Rows reused from a previous manifest must keep all header columns, including names that are not Python
    identifiers (DATE-OBS): losing them once removed the time of 11,837 rows.
    """
    for i in range(3):
        _write(tmp_path / f"SUT_T26_0001_000001_Lev1.0_2026-09-25T00.0{i}.00.000_0972NB03.fits",
               f"2026-09-25T00:0{i}:00")
    first = manifest.build(tmp_path, workers=2, settle_s=0)
    again = manifest.build(tmp_path, workers=2, previous=first, settle_s=0)
    assert again.attrs["reused"] == 3 and again.attrs["read"] == 0
    assert again["DATE-OBS"].notna().all() and again.t.notna().all()
    assert (again.sort_values("file").t.values == first.sort_values("file").t.values).all()
    assert (again.sort_values("file").sha256.values == first.sort_values("file").sha256.values).all()


def test_files_still_being_written_are_skipped(tmp_path):
    _write(tmp_path / "SUT_T26_0001_000001_Lev1.0_2026-09-25T00.00.00.000_0972NB03.fits", "2026-09-25T00:00:00")
    assert len(manifest.build(tmp_path, workers=1, settle_s=3600)) == 0


def test_span_reads_only_inside_and_keeps_the_rest(tmp_path):
    """A data set's pipeline indexes only its own time span; rows of other spans must survive untouched."""
    import pandas as pd
    for day in (20, 25):
        _write(tmp_path / f"SUT_T26_0001_000001_Lev1.0_2026-09-{day}T00.00.00.000_0972NB03.fits", f"2026-09-{day}T00:00:00")
    first = manifest.build(tmp_path, workers=2, settle_s=0, span=(pd.Timestamp("2026-09-19"), pd.Timestamp("2026-09-21")))
    assert list(first.t.dt.day) == [20]
    both = manifest.build(tmp_path, workers=2, previous=first, settle_s=0,
                          span=(pd.Timestamp("2026-09-24"), pd.Timestamp("2026-09-26")))
    assert sorted(both.t.dt.day) == [20, 25] and both.attrs.get("read") == 1
    # nothing new inside a span: the archive is returned unchanged
    again = manifest.build(tmp_path, workers=2, previous=both, settle_s=0,
                           span=(pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02")))
    assert sorted(again.t.dt.day) == [20, 25]
