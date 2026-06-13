#!/bin/bash
# =============================================================================
#  run-msips.sh — Script HPC pour le notebook MS-IPS (NODE21 + DDR + RSNA)
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
#    ~/ips-project/results/ms-ips-op/
#      ├── ms-ips/node21/   → *.pt, *_history.json, *_experiment.csv, confusion_matrix.png
#      ├── ms-ips/ddr/      → *.pt, *_history.json, *_experiment.csv, saliency_ddr/
#      └── ms-ips/rsna/     → *.pt, *_history.json, *_experiment.csv, saliency_rsna/
# =============================================================================

#SBATCH --job-name=msips_experiment
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

NOTEBOOK="ms-ips-op.ipynb"

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
mkdir -p "$RESULTS_DIR/ms-ips/node21"
mkdir -p "$RESULTS_DIR/ms-ips/ddr"
mkdir -p "$RESULTS_DIR/ms-ips/ddr/saliency_ddr"
mkdir -p "$RESULTS_DIR/ms-ips/rsna"
mkdir -p "$RESULTS_DIR/ms-ips/rsna/saliency_rsna"

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
#  codecarbon est installé ici — inutile de le réinstaller depuis le notebook
# =============================================================================
VENV_DIR="$WORK_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[SETUP] Création du virtualenv..."
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "[SETUP] Installation des dépendances..."
pip install --upgrade pip --quiet
pip uninstall -y torch torchvision 2>/dev/null || true

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

# ── 1. Supprimer toutes les lignes !pip install (magics Jupyter) ──────────────
#    codecarbon et toute autre dépendance sont déjà installés par le bash
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
    "'/kaggle/working/ms-ips/node21'",
    f"'{results_dir}/ms-ips/node21'")
src = src.replace(
    '"/kaggle/working/ms-ips/node21"',
    f'"{results_dir}/ms-ips/node21"')

# ── 5. SAVE_PATH ddr ─────────────────────────────────────────────────────────
src = src.replace(
    "'/kaggle/working/ms-ips/ddr'",
    f"'{results_dir}/ms-ips/ddr'")
src = src.replace(
    '"/kaggle/working/ms-ips/ddr"',
    f'"{results_dir}/ms-ips/ddr"')

# ── 6. SAVE_PATH rsna ────────────────────────────────────────────────────────
src = src.replace(
    "'/kaggle/working/ms-ips/rsna'",
    f"'{results_dir}/ms-ips/rsna'")
src = src.replace(
    '"/kaggle/working/ms-ips/rsna"',
    f'"{results_dir}/ms-ips/rsna"')

# ── 7. /kaggle/working générique ─────────────────────────────────────────────
src = src.replace("'/kaggle/working'", f"'{wk}'")
src = src.replace('"/kaggle/working"', f'"{wk}"')
src = re.sub(r"(['\"])/kaggle/working", lambda m: m.group(1) + wk, src)

# ── 8. log_experiment — filet de sécurité ────────────────────────────────────
src = re.sub(
    r"log_experiment\(dir_path\s*=\s*['\"][^'\"]*['\"]",
    f"log_experiment(dir_path='{results_dir}/ms-ips'",
    src)

# ── 9. saliency_ddr / saliency_rsna ──────────────────────────────────────────
src = re.sub(
    r"save_dir\s*=\s*['\"][^'\"]*saliency_ddr['\"]",
    f"save_dir='{results_dir}/ms-ips/ddr/saliency_ddr'",
    src)
src = re.sub(
    r"save_dir\s*=\s*['\"][^'\"]*saliency_rsna['\"]",
    f"save_dir='{results_dir}/ms-ips/rsna/saliency_rsna'",
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
print(f"  ms-ips/node21           → {results_dir}/ms-ips/node21/")
print(f"  ms-ips/ddr              → {results_dir}/ms-ips/ddr/")
print(f"  ms-ips/rsna             → {results_dir}/ms-ips/rsna/")
print(f"  saliency_ddr            → {results_dir}/ms-ips/ddr/saliency_ddr/")
print(f"  saliency_rsna           → {results_dir}/ms-ips/rsna/saliency_rsna/")
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
        DDIR="$RESULTS_DIR/ms-ips/$DATASET"
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
        SDIR="$DDIR/saliency_${DATASET}"
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
    echo "Log : $WORK_DIR/logs/${SLURM_JOB_NAME:-msips}_${SLURM_JOB_ID:-0}.err"
fi

deactivate
exit $EXIT_CODE
