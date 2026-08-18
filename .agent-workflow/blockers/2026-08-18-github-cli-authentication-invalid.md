# GitHub CLI authentication invalid during publish preflight

## Context and intended action

On 2026-08-18, the repository publish workflow was preparing to create a branch,
commit the weighted-ensemble implementation, push it to GitHub, and open a draft
pull request for the existing Streamlit Community Cloud deployment.

## Observable symptom

`gh auth status` reported that the stored credentials for both configured GitHub
accounts were invalid and recommended running `gh auth login -h github.com`.

## Impact

Local implementation, tests, training, and Streamlit verification can continue.
GitHub API operations and the final push/PR step may fail until authentication is
restored.

## Likely cause

The locally cached GitHub CLI tokens have expired, been revoked, or are otherwise
no longer accepted by GitHub.

## Troubleshooting and results

- Confirmed that GitHub CLI is installed (`gh` 2.96.0).
- Confirmed the repository has an `origin` remote pointing to GitHub.
- Ran `gh auth status`; both saved accounts failed authentication.

## Workaround or remaining limitation

Git's separate credential-manager session remained valid, so the feature branch was
successfully pushed with `git push -u origin agent/weighted-hybrid-ensemble`. GitHub
CLI API operations still require `gh auth login -h github.com` before they can be used
as a fallback for pull-request creation. Do not store credentials in this incident note.

## Prevention

- Run `gh auth status` at the beginning of future publish workflows.
- Re-authenticate before starting a time-sensitive deployment when the status check
  reports an invalid token.
