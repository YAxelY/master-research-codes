1. Create your new branch
   
   ```bash
   git checkout -b ipsBranch
   ```

2. Add the repo temporarily

```bash
git remote add ipsRepo https://github.com/benbergner/ips.git
git fetch ipsRepo
```

---

### 3. Replace your working tree with their files (no history)

```bash
git rm -rf .
git checkout ipsRepo/main -- .
```

### 4. Commit as YOUR code

```bash
git commit -m "Imported ips repo content as snapshot"
```

5. Clean up

```bash
git remote remove ipsRepo
```
