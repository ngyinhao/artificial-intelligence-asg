# Deployment push rejected by concurrent main update

- **Context / intended action:** Push the verified ARUF implementation to the GitHub `main` branch that drives the existing Streamlit deployment.
- **Symptom:** Git rejected `main -> main` with `fetch first` because the remote contained work absent from the local branch.
- **Impact:** Deployment did not begin, but no remote or local work was overwritten.
- **Confirmed cause:** Another active session merged a fixed weighted-ensemble implementation after this task began. It overlapped the adaptive implementation in training, inference, UI, tests, metrics, and documentation.
- **Troubleshooting plan:** Fetch the current remote branch, inspect its commits and file overlap, then rebase the ARUF commit onto it. Resolve only genuine overlaps while preserving the newest report content.
- **Tool-version detail:** This Git build rejected `git restore --source=:3` and
  `--source=:2` with `fatal: could not resolve`. The supported conflict-stage form is
  `git restore --theirs` / `git restore --ours`; the rejected command made no changes.
- **Workaround:** Use a non-destructive fetch-and-rebase workflow followed by complete verification and a normal fast-forward push. Never force-push over concurrent work.
- **Prevention:** When multiple sessions share a repository, fetch immediately before the final push and expect report/documentation files to require a last overlap check.
