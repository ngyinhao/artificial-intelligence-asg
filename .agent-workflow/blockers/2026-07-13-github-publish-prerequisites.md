# GitHub publish prerequisites unavailable

## Context and intended action

On 2026-07-13, the repository was being prepared for its first commit and push to GitHub before a Streamlit deployment.

## Observable symptoms

- `git status -sb` reported `No commits yet on main`.
- `git remote -v` returned no configured remotes.
- `gh auth status` reported that the active GitHub account token was invalid and recommended `gh auth login -h github.com`.
- The first `git add --all` attempt failed with `Unable to create '.git/index.lock': Permission denied` because the managed workspace exposed `.git` as read-only inside the default sandbox.
- The first `git commit` attempt failed with `Author identity unknown`; neither `user.name` nor `user.email` was configured for this repository or globally.
- The first push to the newly configured `origin` was rejected with `main -> main (fetch first)` because the GitHub repository already contained commits unrelated to the local root commit.
- Merging the unrelated histories produced the expected add/add conflict in `.gitignore`; the remote contained only GitHub's generic Python ignore template, while the local file contained project-specific exclusions for raw narratives, processed data, trained models, caches, and Streamlit secrets.

## Impact

A local initial commit can be created, but no push or GitHub-backed Streamlit Community Cloud deployment can complete until a remote repository exists and GitHub authentication is restored.

## Cause

This checkout was initialized locally but had never been committed or connected to a remote, and the cached GitHub CLI credential had expired or been revoked.

## Troubleshooting and results

- Confirmed GitHub CLI is installed (`gh` 2.96.0).
- Confirmed the checkout is a valid Git repository on `main`.
- Confirmed there is no remote URL to reuse.
- Retried Git metadata writes through the approved escalated Git command path rather than changing filesystem permissions or bypassing the sandbox.
- Kept all files staged after the author-identity failure; no partial commit was created.
- Avoided a force push; fetch and inspect the remote history before reconciling the two roots.
- Resolve the `.gitignore` conflict in favor of the narrower project-maintained file. This preserves all application-specific privacy and artifact exclusions; the remote generic template did not protect any project path absent from the local rules that is required for this deployment workflow.

## Workaround or remaining limitation

Authenticate with `gh auth login -h github.com`, then either create a GitHub repository or add an existing repository as `origin`. After that, push `main` with upstream tracking. Streamlit Community Cloud can then deploy from the GitHub repository.

Before committing, configure a repository-local author identity with `git config user.name` and `git config user.email`. Prefer repository-local configuration so the workflow does not unexpectedly alter the user's global Git settings.

## Future prevention

- Validate `gh auth status` before starting a publish workflow.
- Initialize new workspaces with a remote URL when the target repository is already known.
- Decide repository name and visibility before the first publish.
