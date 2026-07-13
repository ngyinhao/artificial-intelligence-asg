# PowerShell Get-Content displayed valid UTF-8 as mojibake

- **Context and intended action:** Inspect the Streamlit application before its final smoke test.
- **Observable symptom:** `Get-Content app.py` rendered the compass emoji and em dash as mojibake, making valid UI source appear corrupted.
- **Impact:** The display initially suggested an application defect and could have prompted an unnecessary source edit.
- **Confirmed cause:** The file is valid UTF-8. Reading it with Python using `encoding='utf-8'` and printing Unicode escape sequences confirmed `U+1F9ED` and `U+2014` are stored correctly. The problem is the PowerShell console decoding/rendering path, not the repository file.
- **Troubleshooting result:** Ripgrep did not find the apparent mojibake byte sequences, and explicit UTF-8 decoding showed the intended characters.
- **Workaround:** For non-ASCII source validation, explicitly decode as UTF-8 and inspect code points or escaped representations rather than trusting legacy `Get-Content` console rendering.
- **Prevention:** Avoid modifying files solely from mojibake visible in a Windows shell; first distinguish stored bytes from terminal rendering.
