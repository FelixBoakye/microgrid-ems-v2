# Note on HASHES.txt

HASHES.txt in 42b_ucparity_2026-10-23/ and 42b_extras_2026-10-23/ was written at run time.
For this release, local absolute paths inside run logs, PROGRESS.json and RUN_INFO.json were
replaced with <repo> / <home>, and the run-time guard snapshots (guard_hashes_start.txt) were
removed. Hashes for those log/metadata files therefore no longer match. All result CSV, JSON
result, NPZ and figure files are unchanged and match their recorded hashes.
