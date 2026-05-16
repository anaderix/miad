# FlowMatching git history cleanup

**Date:** 2026-05-12

## Diagnosis

```
~/FlowMatching/.git    957 MB
~/FlowMatching files   86 MB (no experiments/ in checkout)

experiments/* blobs in history: 1057 MB across 286 objects (≈100% of pack)
experiments/ in HEAD:           NO (already deleted)
experiments/ in .gitignore:      NO
```

History contains:
- 18+ training checkpoints (`*.pth`, ~12.6 MB each) under `experiments/*/runs/.../checkpoint_final.pth`
- 100 MB `experiments/output_dmf_ablation.log`
- 38 MB `experiments/output_dmf_dm_515_30000.log`
- Many smaller artifacts under `experiments/experiments/runs/friction_sweep/*`

These were committed at some point and later removed from the tree but **not from history**.

## Fix — path-rewrite (preferred over squash)

**Squash** would compress commit log but doesn't remove specific paths cleanly; **path-rewrite via git-filter-repo** removes `experiments/` from every reachable commit, then GC reclaims the space.

### Recipe

```bash
cd ~/FlowMatching

# 1. backup important state (skip if no remote, or already pushed)
git push origin --all  # push current branches
git push origin --tags # push tags
# (optional) clone a backup: git clone --mirror . ~/FlowMatching.backup.git

# 2. install git-filter-repo (single Python script, no system deps)
pip install --user git-filter-repo
# or pull the standalone script:
# curl -fsSL https://raw.githubusercontent.com/newren/git-filter-repo/main/git-filter-repo \
#     -o ~/.local/bin/git-filter-repo && chmod +x ~/.local/bin/git-filter-repo

# 3. rewrite history to drop experiments/ from every commit
git filter-repo --path experiments --invert-paths --force

# 4. confirm size dropped
du -sh .git  # should be ~50-100 MB
```

`git filter-repo` automatically runs gc/repack and intentionally removes the `origin` remote (safety against accidental push of rewritten history). To resume pushing:

```bash
git remote add origin <your-remote-url>
git push origin --all --force      # warn collaborators first!
git push origin --tags --force
```

### Without internet / system Python

`git filter-branch` (slower, in stdlib):

```bash
git filter-branch --force --index-filter \
  'git rm -rf --cached --ignore-unmatch experiments' \
  --prune-empty --tag-name-filter cat -- --all

# clean refs and GC
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

### Alternative: full-history squash (different intent)

If the goal is also to flatten the commit log (lose history granularity):

```bash
git checkout --orphan flat
git add -A
git commit -m "Squashed history of FlowMatching <date>"
git branch -D main && git branch -m main
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

Note: squash works *only* if the squashed snapshot also drops `experiments/` (current tree already does). It's a heavier hammer; path-rewrite is preferable when commit log has value.

## Prevention

Add to `.gitignore`:

```gitignore
experiments/
*.pth
*.ckpt
*.log
runs/
checkpoints/
training_artifacts/
wandb/
```

For training artifacts, recommend one of:
- DVC (`dvc add experiments/`)
- Plain rsync to a separate machine / S3
- Git LFS if absolutely must stay in repo

## After-cleanup checks

```bash
cd ~/FlowMatching
git status                                            # working tree should be clean
du -sh .git                                           # expect 50-100 MB
git rev-list --objects --all | wc -l                  # significantly fewer
git log --oneline --all | head                        # history should be intact (minus removed paths)
git verify-pack -v .git/objects/pack/*.idx | sort -k3 -n | tail # largest objects now non-experiment
```

## Risks

1. **Force-push needed** if remote has same commits — break for collaborators (they must re-clone).
2. **All commit hashes change** after rewrite — old links / PRs become orphaned.
3. **No undo without a backup** — keep `~/FlowMatching.backup.git` until satisfied.

## Why not just GC?

`git gc --aggressive --prune=now` won't reclaim space because the experiments/* blobs are **still reachable** from old commits in `main`. They're not orphaned — they're history. Only path-rewrite (which makes new commits that don't reference those blobs) can orphan them and let GC remove them.
