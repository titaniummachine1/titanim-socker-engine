# This is NOT where Titanium is developed

This folder is the `main` branch: the **old monolith**, a single 2020-line
`scripts/build_titanium.py` with no `titanium/` package.

## The modules are here instead

```
..\titanim-socker-engine-wallbounce\scripts\titanium\
```

That folder is a **git worktree of this repo** on branch
`wallbounce-fix-on-champion`. Same repository, different branch, checked out
at the same time — which is why the code appears to exist twice.

It is where the live champion is built, and where `constants.py` holds the
measured physics (gravity 17.0, ball radius 0.25, and the rest).

## Do not delete either folder

`titanim-socker-engine-wallbounce\.git` is a *file* pointing into
`titanim-socker-engine\.git\worktrees\`. Deleting this folder breaks the
worktree; deleting the worktree folder without `git worktree remove` leaves
git in a broken state.

To end up with one folder: merge `wallbounce-fix-on-champion` into `main`,
then `git worktree remove ..\titanim-socker-engine-wallbounce`. Both trees
have uncommitted work right now, so review before doing it.

See `..\WHERE_IS_WHAT.md` for the full layout.
