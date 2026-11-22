# Git history cleanup — removing admission identifiers

**Status: prepared, NOT yet run.** This document is the runbook for scrubbing
MIMIC-IV admission identifiers (`hadm_id`) that were committed to git history in
earlier commits (e.g. an old `results/stage3_validation_controls_summary.json`
that carried a 30-identifier sample list, and a few IDs in session notes / a
docstring / a test). The current working tree has already been scrubbed; this
job rewrites the **historical** commits where the identifiers still live.

> **This is destructive and rewrites every commit hash.** It requires a
> force-push and a coordinated re-clone by every collaborator. Do not run any
> step until the whole team has agreed a window. Read all of it first.

## What gets replaced

A total of **33 distinct identifiers** appear across the repository history.
Every occurrence is replaced with the literal placeholder `<hadm_id>`. The
replacements are applied with word-safe regexes so an identifier that happens to
be a substring of a longer number or a decimal fraction is never touched, and
**SLURM job IDs, the project name/date, and citation IDs are deliberately
excluded** (they are not patient data).

The replacement list is held in a local file, **`id_replacements.txt`, which
must never be committed** (it is the one place the plaintext identifiers still
sit). It is kept outside the repository working tree. Each line has the form:

```
regex:(?<![\d.])2XXXXXXX(?![\d.])==><hadm_id>
```

(the `2XXXXXXX` above is an illustration, not a real value). You can either use
the pre-generated 33-line file (ask the person who prepared this runbook for its
local path) or regenerate it yourself with the scan in the next step — the two
are identical.

### Regenerate the replacement list (optional, self-serve)

Run from a clean checkout; writes the file **outside** the repo. This scans all
history blobs for standalone 8-digit integers in the MIMIC id ranges
(`10_000_000`–`29_999_999`), drops anything on a line about a SLURM job plus the
project-name and citation values, and emits one word-safe regex per identifier:

```bash
python3 - "$HOME" <<'PY'
import re, subprocess, sys
pat   = re.compile(rb'(?<![\d.])\d{8}(?![\d.])')
jobkw = re.compile(rb'job|slurm|SLURM_JOB|%j', re.I)
EXCLUDE = {20260401, 12085305, 15700175}   # project name/date, a cited article id, a known job id
oids = [l.split()[1] for l in subprocess.run(
    ["git","cat-file","--batch-all-objects","--batch-check=%(objecttype) %(objectname)"],
    capture_output=True, text=True).stdout.splitlines() if l.startswith("blob")]
ids, proc = set(), subprocess.Popen(["git","cat-file","--batch"],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE)
for oid in oids:
    proc.stdin.write((oid+"\n").encode()); proc.stdin.flush()
    h = proc.stdout.readline().split()
    if len(h) < 3: continue
    data = proc.stdout.read(int(h[2])); proc.stdout.read(1)
    if b'\x00' in data[:1000]: continue
    prev_job = False
    for line in data.splitlines():
        for m in pat.finditer(line):
            n = int(m.group())
            if 10_000_000 <= n <= 29_999_999 and n not in EXCLUDE \
               and not jobkw.search(line) and not prev_job:
                ids.add(n)
        prev_job = bool(jobkw.search(line))
proc.stdin.close()
with open(sys.argv[1] + "/id_replacements.txt", "w") as f:
    for n in sorted(ids):
        f.write(f"regex:(?<![\\d.]){n}(?![\\d.])==><hadm_id>\n")
print(f"wrote {len(ids)} replacements to {sys.argv[1]}/id_replacements.txt  (do NOT commit)")
PY
```

Expected output: `wrote 33 replacements`.

## Prerequisites

```bash
pip install git-filter-repo       # https://github.com/newren/git-filter-repo
git filter-repo --version          # confirm it is on PATH
```

Also make sure the working-tree scrub is already committed (the tip commit must
be identifier-free) **before** rewriting history, so the rewrite only has to fix
older commits:

```bash
git status                         # clean working tree
python -m scripts.check_results_no_identifiers results/*.json models/*.json   # must pass
```

## Step 0 — back up first (non-negotiable)

```bash
# A full mirror backup, kept off to the side. If anything goes wrong, this is
# the only way back.
git clone --mirror . ../readmission-audit-backup.git
# And a local annotated tag on the current tip, in case you only need a soft undo.
git tag pre-history-cleanup-backup
```

## Step 1 — place the replacement list outside the repo

Put `id_replacements.txt` (pre-generated or regenerated above) at a path
**outside** the working tree, e.g. `$HOME/id_replacements.txt`. Confirm it is not
tracked and cannot be committed:

```bash
test -f "$HOME/id_replacements.txt" && echo present
git check-ignore -v "$HOME/id_replacements.txt" 2>/dev/null || echo "outside repo — cannot be committed"
```

## Step 2 — rewrite history

```bash
git filter-repo --replace-text "$HOME/id_replacements.txt"
```

`git filter-repo` rewrites every commit, replacing each identifier with
`<hadm_id>` in every blob and commit message across all branches and tags.

## Step 3 — verify

```bash
# No identifier should survive anywhere in the new history.
python3 - "$HOME" <<'PY'
import re, subprocess, sys
pat = re.compile(rb'(?<![\d.])\d{8}(?![\d.])')
jobkw = re.compile(rb'job|slurm|SLURM_JOB|%j', re.I)
EXCLUDE = {20260401, 12085305, 15700175}
oids = [l.split()[1] for l in subprocess.run(
    ["git","cat-file","--batch-all-objects","--batch-check=%(objecttype) %(objectname)"],
    capture_output=True, text=True).stdout.splitlines() if l.startswith("blob")]
proc = subprocess.Popen(["git","cat-file","--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
hits = 0
for oid in oids:
    proc.stdin.write((oid+"\n").encode()); proc.stdin.flush()
    h = proc.stdout.readline().split()
    if len(h) < 3: continue
    data = proc.stdout.read(int(h[2])); proc.stdout.read(1)
    if b'\x00' in data[:1000]: continue
    prev_job = False
    for line in data.splitlines():
        for m in pat.finditer(line):
            n = int(m.group())
            if 10_000_000 <= n <= 29_999_999 and n not in EXCLUDE \
               and not jobkw.search(line) and not prev_job:
                hits += 1
        prev_job = bool(jobkw.search(line))
print("remaining identifier hits in history:", hits)   # must be 0
PY
```

## Step 4 — force-push the rewritten history

`git filter-repo` removes the `origin` remote on purpose. Re-add it and push all
refs with lease protection:

```bash
git remote add origin <REPO_URL>
git push --force-with-lease origin --all
git push --force-with-lease origin --tags
```

## Step 5 — what every collaborator must do afterwards

Rewriting history changes every commit hash, so existing clones are now
incompatible and still contain the identifiers.

1. **Stop pushing from any old clone immediately.** A push from a pre-rewrite
   clone would re-introduce the identifiers into history. Announce the rewrite
   before doing it.
2. **Re-clone fresh:**
   ```bash
   cd .. && rm -rf <old-clone-dir>
   git clone <REPO_URL>
   ```
   Do not `git pull` an old clone — delete it and clone anew.
3. **Re-apply any in-flight local work** by cherry-picking patches exported from
   the old clone (`git format-patch`), not by merging the old branch.
4. **Open pull requests and forks** still point at the old history. Close and
   re-open PRs from fresh branches; ask fork owners to re-fork.
5. **Caches:** ask the git host (e.g. GitHub support) to run GC / drop stale
   refs if the old objects remain reachable via cached views or PR refs.

## Step 6 — clean up the local secret

```bash
shred -u "$HOME/id_replacements.txt"   # or: rm -P on macOS
rm -rf ../readmission-audit-backup.git # only once the rewrite is confirmed good
```

## Scope note

This runbook covers git history only. The PhysioNet / MIMIC-IV-Note data use
agreement breach of having committed identifiers should also be assessed
separately (who could have cloned the repo while the identifiers were present,
and whether the host retains cached copies). The technical scrub does not, by
itself, close that exposure.
