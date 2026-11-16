#!/bin/bash
# =============================================================================
#  run-ips-suite-v3.sh : cluster script for ips-suite-v3.ipynb (DDR and RSNA)
#
#  Needs this script, the notebook and kaggle.json in the same folder.
#
#  The argument is a list of run names and/or GROUP names, comma separated.
#  Groups: core, ablation, mstudy, patch, base, ms, lowres, hires, lowdata,
#  misc, all. Every model uses at most two patch sizes.
#  The full list with exact costs:  IPS_MODE=list bash run-ips-suite-v3.sh
#
#  ONE CONFIGURATION PER JOB is the fastest: use submit-v3.sh, which
#  submits one job per configuration of a group:
#
#    bash submit-v3.sh core 42        # 22 jobs
#    bash submit-v3.sh ablation 42    #  8 jobs
#    bash submit-v3.sh mstudy 42      # 20 jobs
#
#  Without training (minutes):
#    IPS_MODE=bench   sbatch -J v3_bench  run-ips-suite-v3.sh
#    IPS_MODE=eval_v2 sbatch -J v3_evalv2 run-ips-suite-v3.sh
#
#  Dry run, one epoch:
#    IPS_EPOCHS=1 sbatch -J v3_try run-ips-suite-v3.sh ddr_ips64
#
#  A finished run is skipped, an interrupted run resumes at its last epoch:
#  resubmitting the same command is always safe.
#
#  The first DDR job writes the decoded-image cache (about 33 GB for the three
#  splits, a few minutes); the others wait for it and reuse it.
#
#  Outputs, under ips-project/results/ips-suite-v3/ :
#      checkpoints/   <run>_weights.pt  (best epoch on validation)
#      experiments/   <run>_result.json, <run>_lesions_{test,val}.csv, bench.csv
#      predictions/   <run>_val.npz, <run>_test.npz  (per image)
#      history/       codecarbon/<run>/
#  Then:  python aggregate-ips-suite-v3.py ips-project/results/ips-suite-v3
# =============================================================================

#SBATCH --job-name=ips_suite
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=bigpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jerry.lacmou.zeutouo@u-picardie.fr

set -euo pipefail

# =============================================================================
#  CONFIGURATION
# =============================================================================

NOTEBOOK="ips-suite-v3.ipynb"

DATASETS=(
    "samriddhibagchi/ddr-dataset-credits-to-authors"
    "iamtapendu/rsna-pneumonia-processed-dataset"
)

RUN_NAME="${1:-core}"      # run names and/or group names, comma separated
TAG=$(printf '%s' "${RUN_NAME:-all}" | tr ',' '+' | cut -c1-50)

# =============================================================================
#  Chemins cluster
# =============================================================================
WORK_DIR="ips-project"
DATA_ROOT="$WORK_DIR/data/datasets"
WORK_KAGGLE="$WORK_DIR/outputs"
BACKBONE_CACHE="$WORK_DIR/backbone_cache"

NOTEBOOK_NAME="${NOTEBOOK%.ipynb}"
RESULTS_DIR="$WORK_DIR/results/$NOTEBOOK_NAME"

export TORCH_HOME="$BACKBONE_CACHE"
export HF_HOME="$BACKBONE_CACHE"
export HUGGINGFACE_HUB_CACHE="$BACKBONE_CACHE"

# =============================================================================
#  Init
# =============================================================================
echo "========================================="
echo "Job      : ${SLURM_JOB_NAME:-local}  (${SLURM_JOB_ID:-0})"
echo "Node     : $(hostname)"
echo "Started  : $(date)"
echo "Notebook : $NOTEBOOK"
echo "Run      : $TAG   (mode ${IPS_MODE:-train}, seeds ${IPS_SEEDS:-42})"
echo "Epochs   : ${IPS_EPOCHS:-<config>}"
echo "Results  : $RESULTS_DIR"
echo "========================================="

module purge              2>/dev/null || true
module load cuda/12.6     2>/dev/null || true
module load python/3.11.7 2>/dev/null || true

mkdir -p "$WORK_DIR/logs"
mkdir -p "$DATA_ROOT"
mkdir -p "$WORK_KAGGLE"
mkdir -p "$BACKBONE_CACHE"

# Sous-dossiers alignés sur les OUT_* définis dans le notebook
mkdir -p "$RESULTS_DIR/checkpoints"
mkdir -p "$RESULTS_DIR/history"
mkdir -p "$RESULTS_DIR/experiments"
mkdir -p "$RESULTS_DIR/saliency"
mkdir -p "$RESULTS_DIR/predictions"
mkdir -p "$WORK_DIR/cache"

cd "$WORK_DIR"

# =============================================================================
#  Kaggle credentials
# =============================================================================
SCRIPT_DIR="$(pwd)/.."

if [ ! -f "$SCRIPT_DIR/kaggle.json" ]; then
    echo "[ERROR] kaggle.json introuvable dans $SCRIPT_DIR"
    exit 1
fi
export KAGGLE_CONFIG_DIR="$SCRIPT_DIR"
chmod 600 "$SCRIPT_DIR/kaggle.json"
echo "[OK] Kaggle credentials : $SCRIPT_DIR/kaggle.json"

# =============================================================================
#  Préparation partagée, sérialisée
#
#  Si plusieurs jobs sont soumis d'un coup, ils créeraient le virtualenv et
#  téléchargeraient les datasets en même temps. Le premier arrivé prépare,
#  les autres attendent ici puis réutilisent.
#  Un seul job : le verrou est pris et relâché sans rien coûter.
# =============================================================================
echo "[LOCK] Attente du verrou de préparation..."
exec 9>".setup.lock"
flock 9
echo "[LOCK] Obtenu."

# =============================================================================
#  Virtualenv + dépendances
#  codecarbon est installé ici — inutile de le réinstaller depuis le notebook
# =============================================================================
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[SETUP] Création du virtualenv..."
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

if [ ! -f ".deps_ok" ]; then
    echo "[SETUP] Installation des dépendances..."
    pip install --upgrade pip --quiet

    pip install --quiet \
        torch \
        torchvision \
        --index-url https://download.pytorch.org/whl/cu126

    pip install --quiet \
        numpy \
        scikit-learn \
        matplotlib \
        Pillow \
        PyYAML \
        kaggle \
        nbconvert \
        tqdm \
        pandas \
        scipy \
        codecarbon

    touch ".deps_ok"
else
    echo "[SETUP] Dépendances déjà installées."
fi

# =============================================================================
#  Téléchargement des datasets manquants
# =============================================================================
echo "[CHECK] Vérification des datasets..."
for SLUG in "${DATASETS[@]}"; do
    DEST="data/datasets/$SLUG"
    if [ -d "$DEST" ] && [ "$(ls -A "$DEST" 2>/dev/null)" ]; then
        echo "  [OK]      $SLUG"
    else
        echo "  [MISSING] $SLUG — téléchargement..."
        mkdir -p "$DEST"
        if kaggle datasets download -d "$SLUG" -p "$DEST" --unzip; then
            echo "  [OK]      $SLUG téléchargé"
        else
            echo "  [ERROR]   Échec pour $SLUG"
            exit 1
        fi
    fi
done
echo ""

flock -u 9
echo "[LOCK] Relâché."

# =============================================================================
#  Conversion notebook → script Python
#  Un fichier par job : plusieurs jobs ne doivent pas écrire le même.
# =============================================================================
SCRIPT_PY="_run_${NOTEBOOK_NAME}_${TAG}_s${IPS_SEEDS:-42}_${SLURM_JOB_ID:-local}_$$.py"

echo "[CONVERT] Notebook → script Python..."
jupyter nbconvert \
    --to script \
    --output "${SCRIPT_PY%.py}" \
    --output-dir "." \
    "$SCRIPT_DIR/$NOTEBOOK"

echo "[OK] Script généré : $SCRIPT_PY"

# =============================================================================
#  Patch des magics Jupyter + filet de sécurité sur les chemins
# =============================================================================
echo "[PATCH] Nettoyage du script..."

python - << PYEOF
import re

script = "$SCRIPT_PY"
wk     = "$(pwd)/outputs"

with open(script) as f:
    src = f.read()

# ── 1. Supprimer toutes les lignes !pip install (magics Jupyter) ─────────────
#    codecarbon et toute autre dépendance sont déjà installés par le bash
src = re.sub(r'^.*!pip install.*\$', '', src, flags=re.MULTILINE)

# ── 2. get_ipython() laissé par nbconvert ────────────────────────────────────
src = re.sub(r'^(\s*)get_ipython\(\)', r'\1pass  # get_ipython()', src,
             flags=re.MULTILINE)

# ── 3. /kaggle/working — filet de sécurité ───────────────────────────────────
#    Les chemins viennent normalement des variables IPS_* exportées plus bas.
#    Ce remplacement ne sert que si une valeur par défaut Kaggle survivait.
src = src.replace("'/kaggle/working'", f"'{wk}'")
src = src.replace('"/kaggle/working"', f'"{wk}"')
src = re.sub(r"(['\"])/kaggle/working", lambda m: m.group(1) + wk, src)

# ── 4. tqdm.notebook → tqdm ──────────────────────────────────────────────────
src = src.replace("from tqdm.notebook import tqdm", "from tqdm import tqdm")

# ── 5. plt.show() → plt.close('all') ─────────────────────────────────────────
src = re.sub(r'\bplt\.show\(\)', "plt.close('all')", src)

with open(script, "w") as f:
    f.write(src)

print("  !pip install supprimés   (codecarbon déjà installé par bash)")
print(f"  /kaggle/working         → {wk}   (filet de sécurité)")
print("  Patch OK.")
PYEOF

# =============================================================================
#  Chemins et sélection passés au notebook par l'environnement
# =============================================================================
export IPS_WORK="$(pwd)/outputs"
export IPS_OUT_CKPT="$(pwd)/results/$NOTEBOOK_NAME/checkpoints"
export IPS_OUT_HIST="$(pwd)/results/$NOTEBOOK_NAME/history"
export IPS_OUT_EXPE="$(pwd)/results/$NOTEBOOK_NAME/experiments"
export IPS_OUT_SAL="$(pwd)/results/$NOTEBOOK_NAME/saliency"
export IPS_DDR_DIR="$(pwd)/data/datasets/samriddhibagchi/ddr-dataset-credits-to-authors/DDR-dataset"
export IPS_RSNA_DIR="$(pwd)/data/datasets/iamtapendu/rsna-pneumonia-processed-dataset"
export IPS_OUT_PRED="$(pwd)/results/$NOTEBOOK_NAME/predictions"
export IPS_RUNS="$RUN_NAME"
export IPS_EPOCHS="${IPS_EPOCHS:-}"
export IPS_MODE="${IPS_MODE:-train}"
export IPS_SEEDS="${IPS_SEEDS:-42}"
export IPS_CACHE_DIR="${IPS_CACHE_DIR:-$(pwd)/cache}"
export IPS_CACHE="${IPS_CACHE:-ddr}"
export IPS_WORKERS="${IPS_WORKERS:-12}"
export IPS_V2_CKPT="${IPS_V2_CKPT:-$(pwd)/results/ips-suite/checkpoints}"
export IPS_FORCE="${IPS_FORCE:-0}"
export IPS_DDR_GEOMETRY="${IPS_DDR_GEOMETRY:-stretch}"   # stretch (v2) | retina | crop

echo "[ENV] mode=$IPS_MODE runs=$IPS_RUNS seeds=$IPS_SEEDS workers=$IPS_WORKERS cache=$IPS_CACHE_DIR"

echo ""
echo "[ENV] DDR  → $IPS_DDR_DIR"
echo "[ENV] RSNA → $IPS_RSNA_DIR"
for d in "$IPS_DDR_DIR" "$IPS_RSNA_DIR"; do
    if [ ! -d "$d" ]; then
        echo "[ERROR] Répertoire de données introuvable : $d"
        echo "        Arborescence réelle :"
        find "$(pwd)/data/datasets" -maxdepth 4 -type d | head -30
        exit 1
    fi
done
echo "[OK] Les deux jeux de données sont en place."

# =============================================================================
#  Lancement
# =============================================================================
echo ""
echo "[RUN] Démarrage — $(date)"
echo "========================================="

# Repere temporel : le resume ne listera que les fichiers ecrits par ce job,
# ce qui marche aussi bien pour un modele que pour un groupe.
STAMP=".stamp_${SLURM_JOB_ID:-local}_$$"
touch "$STAMP"

set +e
python "$SCRIPT_PY"
EXIT_CODE=$?
set -e

echo "========================================="
echo "Terminé : $(date)"
echo "Exit code : $EXIT_CODE"

# =============================================================================
#  Résumé post-run
# =============================================================================
if [ $EXIT_CODE -eq 0 ]; then
    rm -f "$SCRIPT_PY"          # conservé en cas d'échec, pour déboguer
    echo ""
    echo "[DONE] Succès."
    echo ""
    echo "── Résultats dans $RESULTS_DIR ──"

    RD="$(pwd)/results/$NOTEBOOK_NAME"

    echo ""
    echo "  Checkpoints :"
    find "$RD/checkpoints" -name "*.pt" -newer "$STAMP" 2>/dev/null | sort \
        | while read f; do
            printf "    %-52s  %s MB\n" \
                "$(basename "$f")" "$(du -m "$f" | cut -f1)"
          done

    echo ""
    echo "  Results :"
    find "$RD/experiments" -name "*_result.json" -newer "$STAMP" 2>/dev/null | sort \
        | while read f; do
            python - "$f" << 'PYEOF2'
import json, sys
r = json.load(open(sys.argv[1])); k = r['main_metric']
print(f"    {r['run_name']:28s} best ep {r['best_epoch']:3d}  val {k} {r['val'].get(k, float('nan')):.4f}  "
      f"TEST {k} {r['test'].get(k, float('nan')):.4f}  peak {r['timing']['train_peak_mb']/1000:.2f} GB")
PYEOF2
          done
    echo ""
    echo "  Other files written by this job :"
    find "$RD/experiments" "$RD/predictions" -type f -newer "$STAMP" 2>/dev/null | wc -l

else
    echo ""
    echo "[ERROR] Échec — exit code $EXIT_CODE"
    echo "Log : $WORK_DIR/logs/${SLURM_JOB_NAME:-ips_suite}_${SLURM_JOB_ID:-0}.err"
fi

rm -f "$STAMP"
deactivate
exit $EXIT_CODE
