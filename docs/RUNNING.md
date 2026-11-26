# Running the suite

Three ways, one code. Every result records where it ran; never mix setups in
a table without saying so (the aggregation script lists them and warns).

## 1. Kaggle, automated (our own quick experiments)

**Once per person**

1. A Hugging Face account and a token with *write* access
   (huggingface.co, Settings, Access Tokens).
2. On Kaggle: a verified phone number (needed for GPU and internet), and the
   Hugging Face token as a secret named `HF_TOKEN` (in a notebook: Add-ons,
   Secrets). After the first launch, open the notebook `ips-suite-v3-runner`
   on Kaggle once and tick `HF_TOKEN` in Add-ons, Secrets, so that it is
   attached to that notebook.
3. On your machine, the Kaggle credentials in a profile folder
   `~/.config/ips-suite/kaggle/<name>/`, either the downloaded `kaggle.json`,
   or a new API token in a file `access_token` with the account name in a file
   `username`. Type the token in your own terminal, never in a chat:

   ```sh
   mkdir -p ~/.config/ips-suite/kaggle/axel && chmod 700 ~/.config/ips-suite
   read -rs T && printf '%s' "$T" > ~/.config/ips-suite/kaggle/axel/access_token && unset T
   echo '<kaggle-username>' > ~/.config/ips-suite/kaggle/axel/username
   chmod 600 ~/.config/ips-suite/kaggle/axel/*
   ```
4. On your machine, the Hugging Face token for pulling results:
   `hf auth login` (or `export HF_TOKEN=...` in your shell profile).

Kaggle allows **one account per person**. Several accounts are fine when each
belongs to a different person (you, a co-author, a supervisor) who agreed to
run the jobs; the profile name is that person's.

**Launch**

```sh
python kaggle/launch.py --profile axel --runs ddr_ms66_33,ddr_h66_33 \
    --hub <hf-user>/ips-suite-runs --epochs 50
python kaggle/launch.py --profile axel --status
```

A code check first (one epoch on 200 images per split, results kept apart
on the Hub because the run names are the same but the data is not):

```sh
python kaggle/launch.py --profile axel --runs ddr_ips66,rsna_ips300 \
    --hub <hf-user>/ips-suite-checks --epochs 1 --smoke 200
```

The script commits nothing and needs a committed notebook: the commit id goes
into every result. Each launch is one private notebook run on Kaggle ("Save &
Run All"), so the computer can be switched off. Give each account a different
set of runs: two accounts must never run the same configuration and seed.

A Kaggle session stops after 12 hours; relaunch the same command and the runs
resume from the Hub where they stopped.

**Collect**

```sh
pip install huggingface_hub
python hub/pull_results.py --hub <hf-user>/ips-suite-runs
python aggregate-ips-suite-v3.py hub-results/ips-suite-v3/results
```

## 2. The supervisor's cluster

Unchanged: see the header of `run-ips-suite-v3.sh`. Results are pushed with
git (`git add ips-project/results ips-project/logs`). Optionally, with
`IPS_HUB_REPO` and `HF_TOKEN` set, the cluster uses the same Hub repository.

## 3. A quick check on Kaggle by hand

Open the notebook on Kaggle, keep the settings cell as it is (one epoch, 200
images per split) and run all. This is the check to do before pushing a change
of the notebook.

## Reporting results

- Say where each number comes from: the `setup` column of the aggregate
  (for example `kaggle/Tesla T4`, `slurm/NVIDIA A40`).
- Scores are comparable across setups when the split hashes are equal.
- Times and memory are comparable only within one GPU type.
- Results made before 2026-11-26 (suite v3.0) used one seed per run instead of
  per-epoch seeds; keep them apart from v3.1 results.
