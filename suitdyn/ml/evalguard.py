"""The one-time test evaluation guard.

The test split is sealed (suitdyn.sequences): reading its windows needs unseal=True and a reason, the frame
list must match the sealed hash, and every read is logged in the seal file. On top of that, the test
EVALUATION is tied to the exact models evaluated: the first read records a fingerprint of the model set
(every checkpoint's SHA-256 plus the learning and data-set config hashes). Reading again with the same
models is harmless (same numbers) and allowed; reading with a different model set means the test split
would start to steer choices, so it is refused unless explicitly overridden, and the override is recorded
permanently in the log.
"""
import hashlib
import json
import time
from pathlib import Path

from .. import atomic


def models_fingerprint(runs, config_hashes):
    """runs: [(name, checkpoint_sha256)]; config_hashes: dict of config SHA-256s."""
    payload = json.dumps({"runs": sorted(map(list, runs)), "configs": config_hashes}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def authorize(record_path, fingerprint, allow_new_models=False):
    """Raise PermissionError if the test split was already read with a different model set (unless allowed).
    Returns whether this read changes the model set relative to the first read."""
    p = Path(record_path)
    reads = json.loads(p.read_text())["reads"] if p.exists() else []
    if not reads:
        return False
    first = reads[0]["fingerprint"]
    if fingerprint == first:
        return False
    if not allow_new_models:
        raise PermissionError(
            f"the test split was first read on {reads[0]['at']} with a different model set (fingerprint "
            f"{first[:12]}..., now {fingerprint[:12]}...). Evaluating new models on it would let the test split "
            f"steer the choices. Pass --allow-new-models only if this is intended; it is recorded permanently.")
    return True


def record(record_path, fingerprint, runs, reason, changed):
    p = Path(record_path)
    log = json.loads(p.read_text()) if p.exists() else {"reads": []}
    log["reads"].append({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason, "fingerprint": fingerprint,
                         "runs": [list(r) for r in runs], "models_changed_after_first_read": bool(changed)})
    atomic.write_json(p, log)
    return log
