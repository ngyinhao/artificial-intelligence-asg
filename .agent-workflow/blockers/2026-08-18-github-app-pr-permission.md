# GitHub app could not create the deployment pull request

## Context and intended action

After the tested weighted-ensemble branch was committed and pushed to GitHub, the
connected GitHub app was used to open the required draft pull request into `main`.

## Observable symptom

GitHub returned HTTP 403 with `Resource not accessible by integration` from the
pull-request creation endpoint.

## Impact

The branch and commit are available on GitHub, but no pull request was created and the
main-branch Streamlit Community Cloud deployment has not been triggered.

## Confirmed cause

The installed GitHub integration can read repository data but does not currently have
permission to create pull requests in this repository.

## Troubleshooting and results

- Local Git credentials successfully pushed the feature branch.
- The GitHub app pull-request action failed with HTTP 403.
- GitHub CLI was already known to have invalid cached API tokens, so it was not a
  viable authenticated fallback in the same run.

## Workaround or remaining limitation

Re-authenticate GitHub CLI using `gh auth login -h github.com`, verify with
`gh auth status`, then create the draft pull request from
`agent/weighted-hybrid-ensemble` into `main`. Alternatively, open the branch's GitHub
pull-request URL manually. Merge through the repository's normal review workflow to
trigger the existing Streamlit deployment.

## Prevention

- Verify both Git push credentials and GitHub app pull-request permissions during the
  initial publishing preflight.
- Keep an authenticated GitHub CLI session available as the documented fallback.
