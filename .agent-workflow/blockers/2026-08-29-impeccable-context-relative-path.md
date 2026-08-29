# Impeccable context helper was not repository-local

- **Context:** Run the required Impeccable context loader before a scoped Streamlit formatting change.
- **Intended action:** Execute `.agents/skills/impeccable/scripts/context.mjs` from the repository root.
- **Symptom:** Node returned `MODULE_NOT_FOUND` for the repository-local path.
- **Impact:** The first context-loading attempt failed; no project files were changed by the command.
- **Cause:** The Impeccable skill is installed under the user's agent skills directory, not under this repository's `.agents/skills` directory.
- **Troubleshooting:** Re-read the resolved skill location supplied by the session.
- **Workaround:** Run `context.mjs` using the skill's absolute installed path while keeping the repository as the working directory.
- **Prevention:** Resolve skill helper paths from the active skill catalog instead of assuming a repository-local installation.
