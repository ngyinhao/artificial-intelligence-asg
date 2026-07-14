# Streamlit URL verification blocked by web fetcher — 2026-07-14

## Context

The README needed the deployed Streamlit application URL, but repository metadata did not contain a homepage URL.

## Intended action

Verify likely `streamlit.app` deployment addresses before writing one into the README.

## Observable symptom

Direct opens of candidate `https://*.streamlit.app/` URLs were rejected by the web fetcher as unsafe.

## Impact

The web fetcher could not establish which candidate deployment URL was valid.

## Likely cause

The fetcher's URL safety policy does not allow direct opening of these unindexed candidate subdomains.

## Troubleshooting and workaround

Use a read-only HTTP request from the repository terminal with approved network access, or obtain the exact deployment URL from the user or Streamlit dashboard.

The first PowerShell fallback did not issue network requests because piping directly after a `foreach` statement produced `An empty pipe element is not allowed`. Wrap the `foreach` expression in `$()` before piping, or emit its results without a trailing pipeline.

The corrected command was then terminated by the shell's default 10-second timeout. Set a command timeout long enough to cover the per-request HTTP timeout when probing multiple URLs.

All three inferred `streamlit.app` slugs returned HTTP 404. A subsequent `Invoke-WebRequest` check of the repository-based `share.streamlit.io` URL failed while processing the response with `Object reference not set to an instance of an object`; use `curl` for redirect/header inspection instead.

## Prevention

Store the deployed application URL in the GitHub repository homepage metadata or a repository configuration file so future workflows do not need to discover it externally.
