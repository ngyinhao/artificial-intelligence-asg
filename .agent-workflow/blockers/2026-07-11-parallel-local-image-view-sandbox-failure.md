# Parallel local image inspection failed in Windows sandbox

- **Date:** 2026-07-11
- **Context and intended action:** Inspect four rendered assignment-PDF pages concurrently using the local image viewer.
- **Observable symptom:** The parallel call failed while locating `tmp/pdfs/assignment/page-02.png`, reporting `windows sandbox failed: helper_unknown_error: apply deny-read ACLs`.
- **Impact:** Batch visual verification could not proceed in one call.
- **Likely cause:** A Windows sandbox helper race or ACL conflict when multiple local-image reads start concurrently.
- **Troubleshooting performed:** The pages had already been rendered successfully by `pypdfium2`; failure occurred only in the image-view sandbox layer.
- **Workaround:** Inspect rendered pages sequentially rather than concurrently.
- **Prevention:** Serialize local `view_image` calls on Windows, or update the sandbox helper to handle concurrent deny-read ACL operations safely.
