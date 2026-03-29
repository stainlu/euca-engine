---
name: Git workflow — commit directly to main
description: User prefers direct commits to main, no PRs. Just commit and push. Only two people in the org.
type: feedback
---

Commit directly to main, no feature branches or PRs unless explicitly requested. Just commit and push.

**Why:** Only two people in the organization. PRs add overhead with no review benefit.

**How to apply:** When batch agents finish work in worktrees, merge their changes directly to main instead of creating PRs. For future batch operations, instruct agents to commit to main directly.
