# `grill-with-docs` supporting skills unavailable

- **Date:** 2026-07-11
- **Context and intended action:** The user explicitly invoked the repository-external `grill-with-docs` skill to frame a new machine-learning project and generate Markdown planning documents.
- **Observable symptom:** The skill's complete instructions consist of: `Run a /grilling session, using the /domain-modeling skill.` A recursive search of the available agent and Codex skill roots returned no `SKILL.md` for either `grilling` or `domain-modeling`, and the `grill-with-docs` package contains no additional files.
- **Impact:** The referenced interview and domain-modeling procedures cannot be followed verbatim.
- **Likely cause:** The installed `grill-with-docs` package has undeclared or missing companion-skill dependencies.
- **Troubleshooting performed:** Searched the available skill roots for directories named `grilling` and `domain-modeling`; inspected the full contents of the `grill-with-docs` directory.
- **Workaround:** Continue with a faithful local approximation: inspect the assignment documentation, conduct a structured requirements and domain interview, and create a project brief, ADRs, and glossary in Markdown.
- **Prevention:** Bundle the two supporting skills with `grill-with-docs`, or document their installation source and dependency setup in the skill package.
