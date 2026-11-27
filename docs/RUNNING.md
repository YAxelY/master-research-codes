# Running the suite

Three ways, one code. Every result records where it ran; never mix setups in
a table without saying so (the aggregation script lists them and warns).

## 1. Kaggle, automated (our own quick experiments)

**Once per person**

1. A Hugging Face account and a token with *write* access
   (huggingface.co, Settings, Access Tokens).

2. On Kaggle: a verified phone number (needed for GPU and internet). The
   Hugging Face token needs no manual step: the launcher stores it, from
   `HF_TOKEN` in the account's .env, in a private dataset of that account
   (`ips-suite-hub-token`) and attaches it to each run (`--refresh-token`
   after changing the token). Prefer a fine-grained token limited to the
   `<hf-user>/ips-suite-*` repositories: whoever can read that dataset can
   read the token.

3. On your machine, a file `.env` at the root of this repository (ignored by
   git), read by the scripts with `--env-file .env`; type the tokens in your
   editor, never in a chat:
   
   ```sh
   KAGGLE_USERNAME=<kaggle-username>
   KAGGLE_API_TOKEN=<kaggle API token>
   HF_TOKEN=<hugging face token>
   IPS_HUB_REPO=<hf-user>/ips-suite-runs
   ```

4. On your machine, the Hugging Face token for pulling results (or `HF_TOKEN`
   in `.env`, with `python hub/pull_results.py --env-file .env ...`):
   `hf auth login` (or `export HF_TOKEN=...` in your shell profile).

Kaggle allows **one account per person**. Several accounts are fine when each
belongs to a different person (you, a co-author, a supervisor) who agreed to
run the jobs; each has their own file in `accounts/`.

**Several accounts.** One file per account in `accounts/` (ignored by git),
for example `accounts/axel.env`, `accounts/kerol.env`, each with
`KAGGLE_USERNAME`, `KAGGLE_API_TOKEN` and optionally `IPS_HUB_REPO`. Each of
these Kaggle accounts needs the `HF_TOKEN` secret too. With `--accounts
accounts`, every account receives the same run list and a different share of
it (`IPS_SHARD`), so no run is launched twice; the shares are written to
`kaggle/last-plan.json`.

**Launch**

```sh
python kaggle/launch.py --env-file .env --runs ddr_ms66_33,ddr_h66_33 --epochs 50
python kaggle/launch.py --accounts accounts --runs focus --epochs 50
python kaggle/launch.py --accounts accounts --status
```

A code check first (one epoch on 200 images per split, results kept apart
on the Hub because the run names are the same but the data is not):

```sh
python kaggle/launch.py --env-file .env --runs ddr_ips66,rsna_ips300 \
    --hub <hf-user>/ips-suite-checks --epochs 1 --smoke 200
```

The script commits nothing and needs a committed notebook: the commit id goes
into every result. Each launch is one private notebook run on Kaggle ("Save &
Run All"), so the computer can be switched off. Give each account a different
set of runs: two accounts must never run the same configuration and seed.

A Kaggle session stops after 12 hours; relaunch the same command and the runs
resume from the Hub where they stopped.
Relaunching is always safe: a session claims each run on the Hub before
training it and refreshes the claim every epoch, so another session skips a
run that is being trained. A claim left by a session Kaggle stopped is taken
over after 2 hours without refresh (`IPS_HUB_CLAIM_STALE_MIN`).

**Collect**

```sh
pip install huggingface_hub
python hub/pull_results.py --hub <hf-user>/ips-suite-runs
python aggregate-ips-suite-v3.py hub-results/ips-suite-v3/results
```

**Analyses next to the results (no heavy download).** An analysis listed in
`analysis/registry.json` runs in a second Kaggle notebook,
`ips-suite-v3-analysis`: it downloads from the Hub only what it needs (light
results, or the weights and Grad-CAM maps), runs the analysis script, and
uploads only its outputs to `ips-suite-v3/analysis/<date>_<name>/`.

```sh
python kaggle/launch.py --env-file .env --analysis aggregate
python kaggle/launch.py --env-file .env --analysis aggregate --status
python hub/pull_results.py --env-file .env --hub <hf-user>/ips-suite-runs --analysis all
```

A new analysis is a tested script in the repository plus one entry in the
registry: what it needs, its command, where it writes.

## 2. The supervisor's cluster

Unchanged: see the header of `run-ips-suite-v3.sh`. Results are pushed with
git (`git add ips-project/results ips-project/logs`). Optionally, with
`IPS_HUB_REPO` and `HF_TOKEN` set, the cluster uses the same Hub repository.

## 3. A quick check on Kaggle by hand

Open the notebook on Kaggle, keep the settings cell as it is (one epoch, 200
images per split) and run all. This is the check to do before pushing a change
of the notebook.

## Reporting results

Every number that leaves this repository (a table, a slide, a message, a
paper) is computed by a script from the raw result files: the result JSONs,
the per-image predictions, the histories. Never retyped, never computed by
hand or by an assistant from a summary. The script is
`aggregate-ips-suite-v3.py` (tables in `_aggregate/`), or a dedicated script
kept next to the document it feeds (for example the `make-data.py` of a
presentation). Quote the output file and the code commit with the number.
The aggregation recomputes each stored test score from the predictions and
warns when they differ. Its tests: `python -m pytest tests`.


- Say where each number comes from: the `setup` column of the aggregate
  (for example `kaggle/Tesla T4`, `slurm/NVIDIA A40`).
- Scores are comparable across setups when the split hashes are equal.
- Times and memory are comparable only within one GPU type.
- Results made before 2026-11-26 (suite v3.0) used one seed per run instead of
  per-epoch seeds; keep them apart from v3.1 results.
