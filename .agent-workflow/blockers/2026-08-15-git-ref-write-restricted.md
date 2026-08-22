# Git ref write restricted in managed workspace

- **Context and intended action:** Create `codex/deploy-minilm-streamlit` before publishing the MiniLM deployment changes.
- **Observable symptom:** `git switch -c` failed with `cannot lock ref` and `unable to create directory for .git/refs/heads/...`.
- **Impact:** The working-tree edits remain intact, but the deployment branch was not created in the default sandbox.
- **Likely cause:** The managed permission profile exposes `.git` for reading but not writing.
- **Troubleshooting:** Confirmed the command remained on `main` and that no branch was partially created.
- **Workaround:** Retry the narrowly scoped Git branch command using the approved elevated execution path.
- **Prevention:** In this workspace profile, expect branch, index, commit, and other `.git` mutations to require approved Git execution.

## Recurrence: restaging documentation cleanup (2026-08-22)

After an initial approved staging operation, a later narrow `git add` used to refresh two documentation files failed with `Unable to create '.git/index.lock': Permission denied`. The working-tree fixes were preserved. Re-running the exact file-scoped staging command through the approved elevated Git path is required; an earlier approval does not necessarily make later sandboxed index writes available.

## Recurrence: discarding regenerated binary-only diffs (2026-08-22)

A narrow `git restore --` for two generated model binaries also failed with `Unable to
create '.git/index.lock': Permission denied`. The command made no changes. Even a
worktree-only restore can require index locking in this environment, so rerun the same
explicitly scoped restore through approved elevated Git execution.
