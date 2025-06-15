---
hide:
  - navigation
---

# Contributing

Here's a step-by-step example of how to contribute to `flooder`. 

1. **Fork the Repository**

   On GitHub, click **"Fork"** to create your personal copy of the repository, then 
   clone your fork, e.g., lets call the fork `flooder-devel`.

   ```bash
   https://github.com/rkwitt/flooder-devel.git
   cd flooder-devel
   ```

2. **Add upstream remote**

   ```bash
   git remote add upstream https://github.com/plus-rkwitt/flooder.git   
   ```

3. **Sync your local main**

   ```bash
   git checkout main
   git fetch upstream
   git rebase upstream/main
   git push origin main
   ```

4. **Create a feature branch**

   ```bash
   git checkout -b fix-typos
   ```

5. **Make changes and commit**

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

6. **Push your branch to your fork**

   ```bash
   git push --force-with-lease origin fix-typos
   ```

7. **Open a PR on GitHub**

   * Navigate to your fork on GitHub.
   * Click "Compare & pull request".
   * Submit the pull request to the upstream repository.

   PR's will be reviewed by the main developers of `flooder`, possibly commented, and then merged in case of no conflicts or concerns.

8. **Cleanup**

   ```bash
   git branch -d fix-typos
   git push origin --delete fix-typos
   ```