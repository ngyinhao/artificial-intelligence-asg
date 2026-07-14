# Skill path lookup failure — 2026-07-14

## Context

While preparing the README update and repository push, the workflow required reading the GitHub publishing and session skill instructions.

## Intended action

Read the relevant `SKILL.md` files before modifying or publishing repository changes.

## Observable symptom

The initial lookup failed because it referenced a non-existent cached path under `openai-bundled/superpowers`.

## Impact

The skill instructions were not read on the first attempt; no repository content was changed by that failed command.

## Likely cause

The active skill catalog maps Superpowers to the curated-remote cache, not the bundled cache.

## Troubleshooting and workaround

Use the active skill catalog mapping directly instead of reconstructing a cache path from memory.

## Prevention

Resolve skill paths from the active skill-root mapping before issuing filesystem reads.
