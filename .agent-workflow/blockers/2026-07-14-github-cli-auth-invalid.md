# GitHub CLI authentication invalid — 2026-07-14

## Context

The user requested a README update followed by pushing the repository.

## Intended action

Use the repository's GitHub publish workflow to verify authentication and push the completed commit.

## Observable symptom

`gh auth status` reports that the active GitHub account's stored token is invalid and recommends re-authentication.

## Impact

The GitHub CLI cannot currently authenticate, so pushing and any GitHub-side publish step are blocked.

## Initially suspected cause

The local GitHub CLI credential has expired or been revoked.

## Troubleshooting and workaround

No credential changes were attempted. Re-authenticate with `gh auth login -h github.com`, then rerun `gh auth status` and the publish workflow.

## Recurrence — 2026-07-14

After the user reported completing login, a fresh `gh auth status` check still reported the same invalid active token. The authentication update has not reached this workspace's GitHub CLI credential store.

## Recurrence — 2026-07-14 (second retry)

A second fresh authentication check produced the same invalid-token result. The push remains blocked until the credential is refreshed in the terminal environment used by this task.

## Confirmed cause and successful workaround

The default managed sandbox could not reach GitHub and misleadingly reported the valid keyring token as invalid. Running `gh auth status` through the approved network-access path confirmed the account and required scopes were valid. GitHub metadata queries then succeeded through the same approved path.

## Prevention

Check `gh auth status` before staging or committing changes intended for GitHub publication. If the user can authenticate in the same workspace but the sandbox still reports an invalid token, retry the read-only status check with approved network access before asking for repeated login attempts.

## Recurrence — 2026-08-15

While preparing to publish the MiniLM artifact for the hosted Streamlit application, the default sandbox again reported both configured GitHub CLI tokens as invalid. Based on the previously confirmed sandbox/network behavior, retry `gh auth status` with approved network access before concluding that user re-authentication is required.

The approved-network retry succeeded for both configured accounts and confirmed that the active account retained the required repository scope. No user re-authentication was necessary.
