# Pip dependency installation blocked by sandbox networking

- **Date:** 2026-07-11
- **Context and intended action:** Install the pinned runtime and test dependencies into the project-local Python 3.12 virtual environment.
- **Observable symptom:** Pip repeatedly failed to connect to the package index with Windows socket error 10013 (`access ... forbidden by its access permissions`) and consequently reported no matching distribution for the first pinned dependency.
- **Impact:** Runtime tests and the Streamlit/model stack cannot be executed until dependencies are installed.
- **Confirmed cause:** Outbound package-index access is blocked in the default managed sandbox, not an invalid package version.
- **Workaround:** Rerun the same pinned pip installation through the approved elevated network-access flow.
- **Prevention:** Expect dependency installation to require network approval in this workspace; keep exact dependencies pinned so the approved operation remains narrow and reproducible.
