# Poppler executables not exposed in bundled PDF runtime

- **Date:** 2026-07-11
- **Context and intended action:** Render and inspect `documentation/Assignment Instruction.pdf` as required by the PDF skill.
- **Observable symptom:** After prepending both documented bundled `bin/override` and `bin/fallback` directories to `PATH`, calls to `pdfinfo` and `pdftoppm` both returned `The system cannot find the path specified.`
- **Impact:** The preferred Poppler-based render workflow is unavailable.
- **Likely cause:** The reported bundled dependency paths do not contain, or do not correctly expose, Poppler executables on this Windows installation.
- **Troubleshooting performed:** Loaded the workspace dependency manifest and used the exact reported binary directories.
- **Workaround:** Check the bundled Python environment for `pypdf`, `pdfplumber`, or PyMuPDF and use available libraries for extraction and page rendering. If no renderer is present, extract text and clearly report the visual-verification limitation.
- **Prevention:** Include Poppler in the Windows primary runtime or have the dependency loader return the exact executable paths and availability status.
