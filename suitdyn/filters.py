"""SUIT science filters, from Tripathi et al. 2025 (Sol."""

# name: (central wavelength nm, FWHM nm, what it samples)
FILTERS = {
    "NB01": (214.0, 11.0, "continuum"),
    "NB02": (276.7, 0.4, "continuum (Mg II blue wing)"),
    "NB03": (279.6, 0.4, "Mg II k"),
    "NB04": (280.3, 0.4, "Mg II h"),
    "NB05": (283.2, 0.4, "continuum (Mg II red wing)"),
    "NB06": (300.0, 1.0, "continuum"),
    "NB07": (388.0, 1.0, "CN band"),
    "NB08": (396.85, 0.1, "Ca II H"),
    "BB01": (221.0, 42.0, "Herzberg continuum (200-242 nm)"),
    "BB02": (271.0, 58.0, "Hartley band (242-300 nm)"),
    "BB03": (330.0, 40.0, "Huggins band (300-360 nm)"),
}
ORDER = list(FILTERS)
