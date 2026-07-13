# PowerShell `foreach` statement could not be piped directly

- **Date:** 2026-07-11
- **Context and intended action:** Build validation objects for several Markdown files and pipe the results to `Format-Table`.
- **Observable symptom:** PowerShell reported `An empty pipe element is not allowed` at the pipe following the closing `foreach` brace.
- **Impact:** The first direct-content validation command did not run; no files were modified.
- **Cause:** In this command form, the `foreach` statement output could not be piped directly as written.
- **Workaround:** Assign the loop output first (`$results = foreach (...) { ... }`) and then pipe `$results` to `Format-Table`.
- **Prevention:** Use an explicit result variable or `ForEach-Object` when formatting loop-generated objects in PowerShell commands.
