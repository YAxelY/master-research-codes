#!/bin/bash
# =============================================================================
#  run-hmsips.sh — Script HPC pour le notebook H-MS-IPS (NODE21 + DDR + RSNA)
#
#  Prérequis :
#    - Ce script, le notebook et kaggle.json dans la branche kaggle-notebooks
#
#  Datasets utilisés :
#    - pshikk/node-21-dataset-untampered              (NODE21)
#    - samriddhibagchi/ddr-dataset-credits-to-authors (DDR)
#    - iamtapendu/rsna-pneumonia-processed-dataset    (RSNA)
#
#  Sorties centralisées dans :
#    ~/ips-project/results/hms-ips-op/
#      ├── hmsips/node21/   → *.pt, *_history.json, *_experiment.csv, confusion_matrix.png, saliency/
#      ├── hmsips/ddr/      → *.pt, *_history.json, *_experiment.csv, confusion_matrix.png, saliency/
#      └── hmsips/rsna/     → *.pt, *_history.json, *_experiment.csv, confusion_matrix.png, saliency/
# =============================================================================

#SBATCH --job-name=hmsips_experiment
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

NOTEBOOK="h-ms-ips.ipynb"

DATASETS=(
    "pshikk/node-21-dataset-untampered"
    "samriddhibagchi/ddr-dataset-credits-to-authors"
    "iamtapendu/rsna-pneumonia-processed-dataset"
)

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
echo "Results  : $RESULTS_DIR"
echo "========================================="

module purge              2>/dev/null || true
module load cuda/12.6     2>/dev/null || true
module load python/3.11.7 2>/dev/null || true

mkdir -p "$WORK_DIR/logs"
mkdir -p "$DATA_ROOT"
mkdir -p "$WORK_KAGGLE"
mkdir -p "$BACKBONE_CACHE"

# Sous-dossiers alignés sur les SAVE_PATH définis dans le notebook
mkdir -p "$RESULTS_DIR/hmsips/node21/saliency"
mkdir -p "$RESULTS_DIR/hmsips/ddr/saliency"
mkdir -p "$RESULTS_DIR/hmsips/rsna/saliency"

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
#  Virtualenv + dépendances
# =============================================================================
VENV_DIR="$WORK_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[SETUP] Création du virtualenv..."
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

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
    SimpleITK \
    h5py \
    Pillow \
    PyYAML \
    kaggle \
    nbconvert \
    tqdm \
    pandas \
    scipy \
    codecarbon

# =============================================================================
#  Téléchargement des datasets manquants
# =============================================================================
echo "[CHECK] Vérification des datasets..."
for SLUG in "${DATASETS[@]}"; do
    DEST="$DATA_ROOT/$SLUG"
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

# =============================================================================
#  Conversion notebook → script Python
# =============================================================================
SCRIPT_PY="$WORK_DIR/_run_${NOTEBOOK_NAME}.py"

echo "[CONVERT] Notebook → script Python..."
jupyter nbconvert \
    --to script \
    --output "_run_${NOTEBOOK_NAME}" \
    --output-dir "$WORK_DIR" \
    "$SCRIPT_DIR/$NOTEBOOK"

echo "[OK] Script généré : $SCRIPT_PY"

# =============================================================================
#  Patch des chemins + nettoyage des magics Jupyter
# =============================================================================
echo "[PATCH] Remplacement des chemins..."

python - << PYEOF
import re

script      = "$SCRIPT_PY"
dr          = "$DATA_ROOT"
wk          = "$WORK_KAGGLE"
bc          = "$BACKBONE_CACHE"
results_dir = "$RESULTS_DIR"

with open(script) as f:
    src = f.read()

# ── 1. Supprimer toutes les lignes !pip install ───────────────────────────────
src = re.sub(r'^.*!pip install.*$', '', src, flags=re.MULTILINE)

# ── 2. Datasets ──────────────────────────────────────────────────────────────
for slug in [
    "pshikk/node-21-dataset-untampered",
    "samriddhibagchi/ddr-dataset-credits-to-authors/DDR-dataset",
    "samriddhibagchi/ddr-dataset-credits-to-authors",
    "iamtapendu/rsna-pneumonia-processed-dataset",
]:
    src = src.replace(
        f"'/kaggle/input/datasets/{slug}'", f"'{dr}/{slug}'")
    src = src.replace(
        f'"/kaggle/input/datasets/{slug}"', f'"{dr}/{slug}"')

# Filet de sécurité
src = re.sub(
    r"(['\"])/kaggle/input/datasets/",
    lambda m: m.group(1) + dr + "/",
    src)

# ── 3. backbone_cache ────────────────────────────────────────────────────────
src = src.replace("'/kaggle/working/backbone_cache'", f"'{bc}'")
src = src.replace('"/kaggle/working/backbone_cache"', f'"{bc}"')

# ── 4. SAVE_PATH node21 ──────────────────────────────────────────────────────
src = src.replace(
    "'/kaggle/working/hmsips/node21'",
    f"'{results_dir}/hmsips/node21'")
src = src.replace(
    '"/kaggle/working/hmsips/node21"',
    f'"{results_dir}/hmsips/node21"')

# ── 5. SAVE_PATH ddr ─────────────────────────────────────────────────────────
src = src.replace(
    "'/kaggle/working/hmsips/ddr'",
    f"'{results_dir}/hmsips/ddr'")
src = src.replace(
    '"/kaggle/working/hmsips/ddr"',
    f'"{results_dir}/hmsips/ddr"')

# ── 6. SAVE_PATH rsna ────────────────────────────────────────────────────────
src = src.replace(
    "'/kaggle/working/hmsips/rsna'",
    f"'{results_dir}/hmsips/rsna'")
src = src.replace(
    '"/kaggle/working/hmsips/rsna"',
    f'"{results_dir}/hmsips/rsna"')

# ── 7. saliency (commun aux trois datasets) ───────────────────────────────────
for dataset in ["node21", "ddr", "rsna"]:
    src = re.sub(
        rf"save_dir\s*=\s*['\"][^'\"]*hmsips/{dataset}/saliency['\"]",
        f"save_dir='{results_dir}/hmsips/{dataset}/saliency'",
        src)

# ── 8. /kaggle/working générique ─────────────────────────────────────────────
src = src.replace("'/kaggle/working'", f"'{wk}'")
src = src.replace('"/kaggle/working"', f'"{wk}"')
src = re.sub(r"(['\"])/kaggle/working", lambda m: m.group(1) + wk, src)

# ── 9. log_experiment — filet de sécurité ────────────────────────────────────
src = re.sub(
    r"log_experiment\(([^)]*)\)",
    lambda m: (
        f"log_experiment(dir_path='{results_dir}/hmsips/node21')"
        if "node21" in m.group(0) else
        f"log_experiment(dir_path='{results_dir}/hmsips/ddr')"
        if "ddr" in m.group(0) else
        f"log_experiment(dir_path='{results_dir}/hmsips/rsna')"
        if "rsna" in m.group(0) else
        f"log_experiment(dir_path='{results_dir}/hmsips')"
    ),
    src)

# ── 10. tqdm.notebook → tqdm ─────────────────────────────────────────────────
src = src.replace("from tqdm.notebook import tqdm", "from tqdm import tqdm")

# ── 11. plt.show() → plt.close('all') ────────────────────────────────────────
src = re.sub(r'\bplt\.show\(\)', "plt.close('all')", src)

with open(script, "w") as f:
    f.write(src)

print(f"  !pip install supprimés   (codecarbon déjà installé par bash)")
print(f"  /kaggle/input/datasets  → {dr}/")
print(f"  /kaggle/working         → {wk}")
print(f"  hmsips/node21           → {results_dir}/hmsips/node21/")
print(f"  hmsips/ddr              → {results_dir}/hmsips/ddr/")
print(f"  hmsips/rsna             → {results_dir}/hmsips/rsna/")
print(f"  saliency/*              → {results_dir}/hmsips/*/saliency/")
print("  Patch OK.")
PYEOF

# =============================================================================
#  Lancement
# =============================================================================
echo ""
echo "[RUN] Démarrage — $(date)"
echo "========================================="

python "$SCRIPT_PY"
EXIT_CODE=$?

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

    for DATASET in node21 ddr rsna; do
        DDIR="$RESULTS_DIR/hmsips/$DATASET"
        echo ""
        echo "  [$DATASET]"

        echo "    Checkpoints :"
        find "$DDIR" -maxdepth 1 -name "*.pt" 2>/dev/null | sort \
            | while read f; do
                printf "      %-55s  %s MB\n" \
                    "$(basename "$f")" "$(du -m "$f" | cut -f1)"
              done

        echo "    History :"
        find "$DDIR" -maxdepth 1 -name "*_history.json" 2>/dev/null | sort \
            | while read f; do echo "      $(basename "$f")"; done

        echo "    Experiments :"
        find "$DDIR" -maxdepth 1 -name "*_experiment.csv" 2>/dev/null | sort \
            | while read f; do echo "      $(basename "$f")"; done

        echo "    Confusion matrix :"
        find "$DDIR" -maxdepth 1 -name "*_confusion_matrix.png" 2>/dev/null | sort \
            | while read f; do echo "      $(basename "$f")"; done

        echo "    Saliency :"
        SDIR="$DDIR/saliency"
        PNG=$(find "$SDIR" -name "*.png" 2>/dev/null | wc -l)
        NPY=$(find "$SDIR" -name "*.npy" 2>/dev/null | wc -l)
        CSV=$(find "$SDIR" -name "*.csv" 2>/dev/null | wc -l)
        echo "      $PNG figures PNG"
        echo "      $NPY fichiers NPY (patches)"
        echo "      $CSV fichiers CSV"
    done

else
    echo ""
    echo "[ERROR] Échec — exit code $EXIT_CODE"
    echo "Log : $WORK_DIR/logs/${SLURM_JOB_NAME:-hmsips}_${SLURM_JOB_ID:-0}.err"
fi

deactivate
exit $EXIT_CODE
