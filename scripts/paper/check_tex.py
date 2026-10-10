"""Check the manuscript before compiling: every macro the text uses is defined, and list what is still open.

    python scripts/paper/check_tex.py
"""
import re
from pathlib import Path

MS = Path(__file__).resolve().parents[2] / "paper" / "manuscript"
num = (MS / "numbers.tex").read_text(encoding="utf-8")
defs = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", num))
tex = (MS / "main.tex").read_text(encoding="utf-8")
body = tex.split(r"\begin{document}", 1)[1]
standard = set("""begin end title author inits fnm snm orcid runningauthor runningtitle address kwd section subsection label
citep citet emph item ref textbf frac sum left right mathcal mathrm centerline includegraphics caption textwidth
textdegree url appendix bibliographystyle bibliography paragraph rm sc input todo pc tbd it bf hline multicolumn
pending textcolor hat mu rho times""".split())
used = set(re.findall(r"\\([A-Za-z]+)", body)) - standard
missing = sorted(u for u in used if u not in defs)
pending = sorted(k for k, v in defs.items() if r"\tbd" in v and k in used)
print(f"number macros used: {len(used & set(defs))}; undefined: {missing or 'none'}")
print(f"numbers not computed yet ({len(pending)}): {pending}")
print(f"open to-do markers in the text: {len(re.findall(r'\\todo\{', body))}")
print(f"conclusions still to check against the v2 results (pending): {len(re.findall(r'\\pending\{', body))}")
