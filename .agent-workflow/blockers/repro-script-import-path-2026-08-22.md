# Repository-local repro script could not import the package

- **Context / intended action:** Run `.agent-workflow/repro_sentence.py` as a tight model-behavior reproduction harness from the repository root.
- **Symptom:** Python raised `ModuleNotFoundError: No module named 'complaint_compass'`.
- **Impact:** The first file-backed reproduction attempt could not reach inference.
- **Cause:** When Python executes a script by path, it places the script's directory (`.agent-workflow`) rather than the repository root first on `sys.path`; this repository is not installed into the active virtual environment as a package.
- **Troubleshooting:** Confirmed the failure with `.\.venv\Scripts\python.exe .agent-workflow\repro_sentence.py`.
- **Workaround:** Set `PYTHONPATH` to the repository root for standalone scripts under `.agent-workflow`, or place a reusable harness under the installed test/package structure.
- **Prevention:** Invoke repository-local diagnostic scripts with an explicit project-root import path, or install the project editable in development environments.

## Recurrence: near-duplicate audit (2026-08-27)

- Running `tools/audit_near_duplicates.py` directly reproduced the same `ModuleNotFoundError` after the script began importing the canonical similarity settings from `complaint_compass.config`.
- The old audit JSON remained on disk, so a failed command followed by an unconditional file read could misleadingly display stale evidence.
- The script now inserts its resolved repository root into `sys.path` before importing the package. Future workflows should also stop chained evidence inspection when the producing command fails and should verify output timestamps or checksums.
