# GitHub pull-request form browser locators were inconsistent

## Context and intended action

The pushed weighted-ensemble branch needed a draft pull request after both the GitHub
app and GitHub CLI paths were unavailable. The authenticated Chrome session was used as
the final publishing fallback.

## Observable symptoms

- The title textbox appeared as `Add a title *` in the DOM snapshot, but filling it by
  that accessible label returned no match.
- After selecting draft mode, the visible split button text and accessible role name
  did not agree, so a role-based click could not locate the submit action.
- The successful submit navigation caused a short browser command timeout even though
  GitHub created the pull request.

## Impact

The browser publishing step required additional state inspection before it could be
completed and verified safely.

## Confirmed cause

GitHub's dynamic pull-request form exposes different names through its rendered DOM,
accessible roles, and split-button state. Navigation can also detach the originating
control before the browser command returns.

## Successful workaround

- Fill the title using its stable `Title` placeholder.
- Fill the description using the `Comment` textbox role.
- Select `Create draft pull request`, return the viewport to the form, and inspect the
  visible DOM.
- Click the visible submit node whose text is `Draft pull request`.
- Treat the resulting `/pull/<number>` URL and visible `Draft` state as the authoritative
  success signal after a navigation-time command timeout.

## Prevention

Inspect the current visible DOM after changing GitHub split-button state, and verify the
resulting URL before retrying a submit action that may already have succeeded.
