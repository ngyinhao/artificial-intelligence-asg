# Tool wrapper name mismatch

- **Context and intended action:** Read the required GitHub publishing skill before preparing a Streamlit deployment.
- **Observable symptom:** The JavaScript orchestration call failed with `TypeError: tools.shell_command is not a function`.
- **Impact:** The first read-only command did not execute; no repository or external state changed.
- **Cause:** This session exposes the command runner as `tools.exec_command`, while an older wrapper name was used.
- **Troubleshooting:** Retried the same read with `tools.exec_command`; it completed successfully.
- **Workaround:** Use `tools.exec_command` for shell operations in this tool environment.
- **Prevention:** Check the currently exposed nested-tool declarations rather than relying on command-wrapper names from prior sessions.
