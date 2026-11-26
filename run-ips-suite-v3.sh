#!/bin/bash
# =============================================================================
#  run-ips-suite-v3.sh : cluster script for the ips_suite package (DDR and RSNA)
#
#  Runs src/ips_suite (python -m ips_suite), the same code as the notebook.
#  Needs this script, src/ and kaggle.json in the same folder.
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
#    bash submit-v3.sh ablation 42    # 36 jobs
#    bash submit-v3.sh mstudy 42      # 54 jobs
#
#  Without training (minutes):
#    IPS_MODE=bench   sbatch -J v3_bench  run-ips-suite-v3.sh
#    IPS_MODE=eval_v2 sbatch -J v3_evalv2 run-ips-suite-v3.sh
#
#  Dry run, one epoch:
#    IPS_EPOCHS=1 sbatch -J v3_try run-ips-suite-v3.sh ddr_ips64
#
#  Your own copy of the data sets (nothing is downloaded for them):
#    IPS_DDR_DIR=/data/DDR-dataset IPS_RSNA_DIR=/data/rsna-pneumonia-processed-dataset \
#        sbatch run-ips-suite-v3.sh ddr_ips66
#  or write the two folders in DDR_DIR_USER / RSNA_DIR_USER below.
#
#  A finished run is skipped, an interrupted run resumes at its last epoch:
#  resubmitting the same command is always safe.
#
#  The first DDR job writes the decoded-image cache (about 33 GB for the three
#  splits at 896 px, about 15 GB more at 1792 px); the others wait for it and
#  reuse it.
#
#  Outputs. ONE FOLDER PER CONFIGURATION (every seed inside it), split in two:
#
#  ips-project/results/ips-suite-v3/      LIGHT, the only folder to push
#      <configuration>/experiments/   <run>_result.json, lesion and Grad-CAM tables
#      <configuration>/history/       <run>_history.json, codecarbon/
#      <configuration>/predictions/   <run>_val.npz, <run>_test.npz  (per image, small)
#      _bench/experiments/bench.csv   (IPS_MODE=bench)
#  ips-project/artifacts/ips-suite-v3/    HEAVY, never pushed (IPS_ARTIFACTS)
#      <configuration>/checkpoints/   <run>_weights.pt, resume state
#      <configuration>/saliency/      <run>_gradcam_test.npz  (about 75 MB per run)
#
#  Also never pushed (all listed in .gitignore): ips-project/cache/ (decoded
#  images, up to 33 GB), ips-project/backbone_cache/ (ImageNet weights),
#  ips-project/outputs/, ips-project/data/, the generated _run_*.py.
#  To push the results of a job:
#      git add ips-project/results ips-project/logs && git commit && git push
#  To free the image cache once every DDR and RSNA job is finished:
#      rm -f ips-project/cache/*.npy
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

# Your own copy of a data set: put its folder here (or give it on the command
# line: IPS_DDR_DIR=/data/DDR-dataset sbatch run-ips-suite-v3.sh ...).
# Left empty, the copy under ips-project/data/datasets/ is used, and
# downloaded from Kaggle if it is missing.
#   DDR : the folder that contains DR_grading/ (and lesion_segmentation/)
#   RSNA: the folder that contains stage2_train_metadata.csv and Training/
DDR_DIR_USER="${IPS_DDR_DIR:-}"
RSNA_DIR_USER="${IPS_RSNA_DIR:-}"
# relative paths are taken from the folder where the job is launched
[ -n "$DDR_DIR_USER" ]  && DDR_DIR_USER="$(realpath -m "$DDR_DIR_USER")"
[ -n "$RSNA_DIR_USER" ] && RSNA_DIR_USER="$(realpath -m "$RSNA_DIR_USER")"

RUN_NAME="${1:-core}"      # run names and/or group names, comma separated
TAG=$(printf '%s' "${RUN_NAME:-all}" | tr ',' '+' | cut -c1-50)

# =============================================================================
#  Chemins cluster
# =============================================================================
WORK_DIR="ips-project"
DATA_ROOT="$WORK_DIR/data/datasets"
WORK_KAGGLE="$WORK_DIR/outputs"
BACKBONE_CACHE="$(pwd)/$WORK_DIR/backbone_cache"   # absolute: the job cds into $WORK_DIR

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
mkdir -p "$RESULTS_DIR"          # the notebook creates one sub-folder per configuration
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
        six \
        nbconvert \
        tqdm \
        pandas \
        scipy \
        codecarbon

    touch ".deps_ok"
else
    echo "[SETUP] Dépendances déjà installées."
fi
# the kaggle client imports six without declaring it: an existing venv may lack it
python -c "import six" 2>/dev/null || pip install --quiet six
[ -n "${IPS_HUB_REPO:-}" ] && { python -c "import huggingface_hub" 2>/dev/null || pip install --quiet huggingface_hub; }

# =============================================================================
#  Téléchargement des datasets manquants
# =============================================================================
echo "[CHECK] Vérification des datasets..."
for SLUG in "${DATASETS[@]}"; do
    DEST="data/datasets/$SLUG"
    case "$SLUG" in
        *ddr*)  OWN="$DDR_DIR_USER" ;;
        *rsna*) OWN="$RSNA_DIR_USER" ;;
        *)      OWN="" ;;
    esac
    if [ -n "$OWN" ]; then
        if [ -d "$OWN" ]; then
            echo "  [OK]      $SLUG : your copy, $OWN"
            continue
        fi
        echo "  [ERROR]   $SLUG : folder not found: $OWN"
        exit 1
    fi
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
#  Chemins et sélection passés au notebook par l'environnement
# =============================================================================
export IPS_WORK="$(pwd)/outputs"
export IPS_RESULTS="$(pwd)/results/$NOTEBOOK_NAME"
export IPS_ARTIFACTS="${IPS_ARTIFACTS:-$(pwd)/artifacts/$NOTEBOOK_NAME}"
export IPS_DDR_DIR="${DDR_DIR_USER:-$(pwd)/data/datasets/samriddhibagchi/ddr-dataset-credits-to-authors/DDR-dataset}"
export IPS_RSNA_DIR="${RSNA_DIR_USER:-$(pwd)/data/datasets/iamtapendu/rsna-pneumonia-processed-dataset}"
export IPS_RUNS="$RUN_NAME"
export IPS_EPOCHS="${IPS_EPOCHS:-}"
export IPS_MODE="${IPS_MODE:-train}"
export IPS_SEEDS="${IPS_SEEDS:-42}"
export IPS_CACHE_DIR="${IPS_CACHE_DIR:-$(pwd)/cache}"
export IPS_CACHE="${IPS_CACHE:-ddr}"
export IPS_WORKERS="${IPS_WORKERS:-12}"
export IPS_V2_CKPT="${IPS_V2_CKPT:-$(pwd)/results/ips-suite/checkpoints}"
export IPS_CODE_COMMIT="${IPS_CODE_COMMIT:-$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null)}"
export IPS_FORCE="${IPS_FORCE:-0}"
export IPS_DDR_GEOMETRY="${IPS_DDR_GEOMETRY:-retina}"    # retina | stretch (v2 input) | crop

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
PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" python -m ips_suite
EXIT_CODE=$?
set -e

echo "========================================="
echo "Terminé : $(date)"
echo "Exit code : $EXIT_CODE"

# =============================================================================
#  Résumé post-run
# =============================================================================
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "[DONE] Succès."
    echo ""
    echo "── Résultats dans $RESULTS_DIR ──"

    RD="$(pwd)/results/$NOTEBOOK_NAME"

    echo ""
    echo "  Checkpoints (in $IPS_ARTIFACTS, not to push) :"
    find "$IPS_ARTIFACTS" -path "*/checkpoints/*" -name "*_weights.pt" -newer "$STAMP" 2>/dev/null | sort \
        | while read f; do
            printf "    %-52s  %s MB\n" \
                "$(basename "$f")" "$(du -m "$f" | cut -f1)"
          done

    echo ""
    echo "  Results :"
    find "$RD" -path "*/experiments/*" -name "*_result.json" -newer "$STAMP" 2>/dev/null | sort \
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
    find "$RD" -type f -newer "$STAMP" 2>/dev/null | wc -l
    echo ""
    echo "  To push only the results:  git add ips-project/results ips-project/logs"

else
    echo ""
    echo "[ERROR] Échec — exit code $EXIT_CODE"
    echo "Log : $WORK_DIR/logs/${SLURM_JOB_NAME:-ips_suite}_${SLURM_JOB_ID:-0}.err"
fi

rm -f "$STAMP"
deactivate
exit $EXIT_CODE
