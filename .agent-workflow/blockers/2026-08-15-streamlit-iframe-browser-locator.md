# Streamlit control requires iframe-scoped browser locator

- **Context and intended action:** Verify the newly deployed MiniLM option in the hosted Streamlit model selector.
- **Observable symptom:** The page snapshot displayed the combobox, but a page-level role locator reported no matches and timed out.
- **Impact:** The first live interaction did not click the selector; the page remained unchanged.
- **Likely cause:** Streamlit Community Cloud renders the application controls inside an iframe, while the page-level locator does not automatically cross that frame boundary.
- **Troubleshooting:** The DOM snapshot explicitly showed the application content nested beneath an `iframe` node.
- **Additional evidence:** A generic `iframe` frame locator was also ambiguous because the page contains both `iframe[title="streamlitApp"]` and a separate `iframe[title="Streamlit Cloud Status"]`.
- **Workaround:** Scope interactive locators specifically with `iframe[title="streamlitApp"]` before selecting controls.
- **Prevention:** For hosted Streamlit browser tests, inspect the initial DOM topology and use frame-scoped locators when the application is embedded.

## Recurrence: cross-origin iframe inspection denied (2026-08-22)

- **Context:** Verify that Streamlit Cloud had deployed the integrated five-model commit
  and changed the visible default from Linear SVM to the weighted ensemble.
- **Symptom:** The outer page snapshot initially exposed the embedded app, but a later
  iframe inspection was denied by browser security policy for the generated
  `*.streamlit.app` origin.
- **Impact:** GitHub `main` and local Streamlit behavior are verified, but the browser
  cannot provide a final cross-origin rendered-state confirmation in this session.
- **Workaround / limitation:** Do not bypass the browser restriction or switch to an
  indirect surface to inspect the same iframe. Use the permitted outer snapshot if it
  updates naturally; otherwise confirm the deployment from the Streamlit UI manually.

## Recurrence: grouped-split deployment verification (2026-08-29)

- The outer snapshot showed the fully loaded rebuilt application, but a page-level `getByLabel("Complaint narrative")` fill timed out with no matches.
- The known `iframe[title="streamlitApp"]` scoping requirement still applies. Use that specific frame locator for the synthetic prediction and comparison-tab checks.
- The iframe-scoped synthetic prediction succeeded and displayed `Mortgage` with `Model confidence` at 95.3%. Switching to the comparison tab then triggered Streamlit Cloud's temporary "taking longer than normal" recovery page for more than 30 seconds. Use a fresh tab for a read-only retry rather than repeatedly interacting with the recovering embed.
