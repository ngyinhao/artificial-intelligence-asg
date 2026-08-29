# Black formatter unavailable in the project environment

- **Context:** Final formatting QA for the similarity-grouped dataset pipeline and tests.
- **Intended action:** Run `python -m black --check` on the changed Python files.
- **Symptom:** Python returned `No module named black`.
- **Impact:** Automated Black conformance could not be checked.
- **Cause:** `requirements-dev.txt` contains pytest tooling but does not include Black.
- **Troubleshooting:** Confirmed the command used the active project virtual environment and inspected the development requirements.
- **Workaround:** Manually wrapped the newly added code, ran the complete test suite, and used `git diff --check` for whitespace validation.
- **Prevention:** Add an agreed formatter to development dependencies and CI only if the project adopts that formatting standard.
