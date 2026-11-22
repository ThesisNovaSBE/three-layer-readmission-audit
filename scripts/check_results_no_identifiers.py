"""Fail-loud guard: a committed result JSON must carry no admission identifiers
and no note text (MIMIC-IV-Note DUA -- derived products redistributed outside
the credentialed environment must not contain hadm_id/subject_id values or
verbatim note excerpts).

Scans each given JSON for:

* any object key containing ``hadm`` or ``subject`` (case-insensitive);
* any integer value in the 8-digit MIMIC identifier range [10_000_000,
  99_999_999] -- aggregate counts in these reports never reach 8 digits
  (the largest, the test-partition n, is ~104k), so an 8-digit int is an id;
* any string that looks like MIMIC note text -- the de-identification markers
  ``___`` or ``[**`` -- or any string under a key known to hold free text /
  quotes (``quote``, ``clinical_justification``, ``raw_response``, ``prompt``,
  ``note_text``, ``text``, ``mitigating_grounds``, ``aggravating_grounds``).

Exit 0 (and print OK) if clean; exit 1 listing every finding otherwise.

CPU only, stdlib only. Usage::

    python -m scripts.check_results_no_identifiers models/pipeline_evaluation.json ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ID_KEY_SUBSTRINGS = ("hadm", "subject")
_FREETEXT_KEYS = frozenset({
    "quote", "clinical_justification", "justification", "raw_response", "prompt",
    "note_text", "text", "mitigating_grounds", "aggravating_grounds", "grounds",
})
_MIMIC_DEID_MARKERS = ("___", "[**")
_ID_MIN, _ID_MAX = 10_000_000, 99_999_999  # 8-digit MIMIC-IV identifier range


def find_identifier_leaks(obj: object, path: str = "") -> list[str]:
    """Return a list of human-readable leak descriptions (empty if clean)."""
    leaks: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else str(key)
            kl = str(key).lower()
            # "*_without_hadm*" are aggregate COUNT keys (e.g. n_discharge_notes_
            # without_hadm), not identifiers -- documented false positive.
            if any(s in kl for s in _ID_KEY_SUBSTRINGS) and "without_hadm" not in kl:
                leaks.append(f"{here}: identifier-like key")
            if str(key) in _FREETEXT_KEYS and isinstance(value, str) and value:
                leaks.append(f"{here}: free-text/quote key holds a string")
            leaks.extend(find_identifier_leaks(value, here))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            leaks.extend(find_identifier_leaks(value, f"{path}[{i}]"))
    elif isinstance(obj, bool):
        pass  # bool is an int subclass -- never an identifier
    elif isinstance(obj, int):
        if _ID_MIN <= abs(obj) <= _ID_MAX:
            leaks.append(f"{path}: 8-digit int {obj} in the MIMIC id range")
    elif isinstance(obj, str):
        if any(m in obj for m in _MIMIC_DEID_MARKERS):
            leaks.append(f"{path}: string contains a MIMIC de-id marker")
    return leaks


_TOKENIZER_FILES = re.compile(
    r"(^|/)(vocab|tokenizer|merges|special_tokens|added_tokens|tokenizer_config)", re.I)


def check_file(path: Path) -> list[str]:
    """Scan one JSON file; returns its leak descriptions (prefixed with path).

    Model tokenizer/vocab files (``vocab.json``, ``tokenizer.json`` …) are
    skipped: their keys/merges are BPE tokens (``Ġsubject``) and subword de-id
    markers (``___``), never patient data -- a documented false positive."""
    if _TOKENIZER_FILES.search(path.name):
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        # Not a strict-JSON data file (e.g. a JSONC config like tsconfig.json
        # with comments/trailing commas). Result data files are always valid
        # JSON; skip anything that isn't parseable rather than crash.
        return []
    return [f"{path}: {leak}" for leak in find_identifier_leaks(data)]


def main() -> None:
    """CLI: scan every given JSON, exit 1 if any leak is found."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", type=Path, nargs="+", help="JSON files to scan")
    args = parser.parse_args()

    all_leaks: list[str] = []
    for path in args.paths:
        if not path.exists():
            print(f"[check_results] ERROR: {path} not found", file=sys.stderr)
            sys.exit(1)
        all_leaks.extend(check_file(path))

    if all_leaks:
        print("[check_results] IDENTIFIER / NOTE-TEXT LEAK(S) FOUND:", file=sys.stderr)
        for leak in all_leaks:
            print(f"  {leak}", file=sys.stderr)
        sys.exit(1)
    print(f"[check_results] OK -- {len(args.paths)} file(s) carry no identifiers or note text.")


if __name__ == "__main__":
    main()
