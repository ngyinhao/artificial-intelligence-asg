# Generic Sites hosting is incompatible with the Streamlit model application

- **Context / intended action:** Publish the completed ARUF-enabled ComplaintCompass application after implementation and verification.
- **Observable constraint:** The available Sites workflow requires a Vite/Cloudflare Worker-compatible build with a generated `dist/server/index.js`. ComplaintCompass is a Python Streamlit application that loads local sklearn, SentenceTransformer, and PyTorch artifacts.
- **Impact:** Publishing through Sites would require replacing the runtime and application architecture, and would not preserve the verified Python inference implementation.
- **Cause:** Sites hosts JavaScript/Cloudflare-compatible applications, while the existing project is already designed and documented for GitHub-backed Streamlit hosting.
- **Workaround:** Preserve the current architecture, push the exact validated revision to the existing GitHub repository, allow the linked Streamlit deployment to rebuild, and verify the public application there.
- **Prevention:** Select deployment tooling based on the application's runtime before beginning a publishing workflow; use Sites for compatible web builds and Streamlit hosting for Python Streamlit applications with local model artifacts.
