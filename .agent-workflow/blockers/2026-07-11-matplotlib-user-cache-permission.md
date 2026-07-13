# Matplotlib could not write its default Windows user cache

- **Date:** 2026-07-11
- **Context and intended action:** Smoke-test the sealed-evaluation CLI after installing dependencies.
- **Observable symptom:** Importing Matplotlib reported `Access is denied` while creating the default cache under the user profile, then created a temporary cache directory.
- **Impact:** Evaluation help still ran, but repeated imports would be slower and multiprocessing behavior could be less reliable.
- **Cause:** The managed filesystem permits workspace writes but not creation of the default user-profile Matplotlib directory.
- **Successful workaround:** Set `MPLCONFIGDIR` to the project's ignored, writable `.cache/matplotlib` path before importing Matplotlib.
- **Prevention:** Route library caches to workspace-local ignored directories in managed Windows environments.
