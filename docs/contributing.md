---
hide:
  - navigation
---

# Contributing

Here's a step-by-step example of how to contribute to `flooder`.

## Fork the repository

   On GitHub, click **"Fork"** to create your personal copy of the repository in your GitHub account, then
   clone your fork. In the following, lets call the fork `flooder-devel`.

   ```bash
   git clone https://github.com/rkwitt/flooder-devel.git
   cd flooder-devel
   ```

## Add upstream remote

   ```bash
   git remote add upstream https://github.com/plus-rkwitt/flooder.git   
   ```

## Sync your local main

   Surely, if you just forked, everything will be in sync (but just to be sure :)

   ```bash
   git checkout main
   git fetch upstream
   git rebase upstream/main
   git push origin main
   ```

## Create a feature branch

   Next, we create a feature branch which will contain our adjustments/enhancements/etc.

   ```bash
   git checkout -b fix-typos
   ```

## Make changes and commit

   Once you are done with your changes, commit.

   ```bash
   git commit -a -m "ENH: Fixed some typos."
   ```

   *What if `upstream/main` divereged in the meantime (e.g., a PR 
   got merged or so)?*

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Fix files in case of conflicts, then add them and continue the rebase.

   ```bash
   git add <file>
   git rebase --continue
   ```

## Push your branch to your fork

   ```bash
   git push --force-with-lease origin fix-typos
   ```

## Open a PR on GitHub

Finally, create a pull request on GitHub.

* Navigate to your fork on GitHub.
* Click "Compare & pull request".
* Submit the pull request to the upstream repository.

   PR's will be reviewed by the main developers of `flooder`, possibly commented, and then merged in case of no conflicts or concerns.

## Cleanup

Once the PR is merged, we cleanup the branch in the forked repo.

   ```bash
   git branch -d fix-typos
   git push origin --delete fix-typos
   ```
