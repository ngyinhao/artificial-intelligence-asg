# Tool wrapper rejected a duplicate JavaScript constant

- **Date:** 2026-07-11
- **Context and intended action:** Update implementation progress and then create the local Python 3.12 virtual environment in one orchestration call.
- **Observable symptom:** The wrapper failed immediately with `SyntaxError: Identifier 'r' has already been declared`.
- **Impact:** Neither the progress update nor the environment command ran; project files were unaffected.
- **Cause:** Two results in the same JavaScript module were assigned to `const r`.
- **Workaround:** Use distinct variable names for every result or avoid assigning results that do not need later access.
- **Prevention:** Name orchestration results by purpose, such as `planResult` and `commandResult`, when composing multiple calls.
