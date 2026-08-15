# Git ref write restricted in managed workspace

- **Context and intended action:** Create `codex/deploy-minilm-streamlit` before publishing the MiniLM deployment changes.
- **Observable symptom:** `git switch -c` failed with `cannot lock ref` and `unable to create directory for .git/refs/heads/...`.
- **Impact:** The working-tree edits remain intact, but the deployment branch was not created in the default sandbox.
- **Likely cause:** The managed permission profile exposes `.git` for reading but not writing.
- **Troubleshooting:** Confirmed the command remained on `main` and that no branch was partially created.
- **Workaround:** Retry the narrowly scoped Git branch command using the approved elevated execution path.
- **Prevention:** In this workspace profile, expect branch, index, commit, and other `.git` mutations to require approved Git execution.
