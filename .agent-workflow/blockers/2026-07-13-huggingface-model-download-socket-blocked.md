# Hugging Face model download blocked by sandbox networking

- **Context and intended action:** Train all ComplaintCompass models, including `sentence-transformers/all-MiniLM-L6-v2`, after preparing the production dataset.
- **Observable symptom:** The Hugging Face client failed on `HEAD` requests with Windows error 10013 (`An attempt was made to access a socket in a way forbidden by its access permissions`). Its retry path then ended with `RuntimeError: Cannot send a request, as the client has been closed.`
- **Impact:** The Naive Bayes and calibrated Linear SVM stages ran, but the MiniLM plus Logistic Regression model could not begin until its pretrained encoder files were available.
- **Likely cause:** The default execution sandbox blocks outbound access to `huggingface.co`; the encoder was not already present in the local Hugging Face cache.
- **Troubleshooting result:** Retrying inside the same restricted process is ineffective because the underlying HTTP client remains unable to open the socket.
- **Workaround:** Rerun the training command with explicitly approved network access so Hugging Face can populate its local cache. Subsequent application inference is local and does not require network access.
- **Prevention:** During a clean setup, download or pre-cache the pinned MiniLM model while network access is available before testing offline training or inference.
