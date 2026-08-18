# Git branch creation blocked by repository metadata permissions

## Context and intended action

On 2026-08-18, the publish workflow attempted to create the feature branch
`agent/weighted-hybrid-ensemble` before editing the weighted-ensemble feature.

## Observable symptom

`git switch -c agent/weighted-hybrid-ensemble` failed with an error saying Git
could not lock the new ref because it could not create the corresponding directory
under `.git/refs/heads/`.

## Impact

The working tree remains on the default branch until the branch operation is rerun
with permission to write repository metadata. Source files were not changed by the
failed command.

## Likely cause

The managed workspace permits reading `.git` but the sandbox does not permit the
write needed to create a branch ref.

## Troubleshooting and result

- Confirmed the current branch was `main` before the attempt.
- Used a normal non-destructive `git switch -c` command; it failed before creating
  the branch.

## Workaround

Rerun the exact branch-creation operation with narrowly scoped elevated permission.

## Prevention

Expect branch, commit, and other `.git` metadata writes to require approval in this
managed workspace even when ordinary workspace source files are writable.
