#!/bin/bash
# =============================================================================
#  run-master2-qmix-standard-ce.sh — Template générique pour notebooks IPS sur cluster HPC
#
#  Prérequis :
#    - Ce script, le notebook et kaggle.json sont dans la branche kaggle-notebooks
#
#  Pour adapter à un autre notebook :
#    1. Changer NOTEBOOK=
#    2. Changer DATASETS=
#
#  Toutes les sorties sont centralisées dans :
#    ~/ips-project/results/<nom-notebook>/
#      ├── checkpoints/   → *.pt
#      ├── history/       → *_history.json
#      ├── experiments/   → *_experiment.csv
#      └── saliency/      → PNG + CSV + NPY
# =============================================================================

#SBATCH --job-name=ips_experiment
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
#  CONFIGURATION — seule section à modifier d'un notebook à l'autre
# =============================================================================

NOTEBOOK="master2-qmix-standard-ce.ipynb"

DATASETS=(
    "samriddhibagchi/ddr-dataset-credits-to-authors"
  
)
EXP_NUM="001" 
# =============================================================================
#  Chemins cluster (ne pas modifier)
# =============================================================================
WORK_DIR="ips-project"
DATA_ROOT="$WORK_DIR/data/datasets"        # remplace /kaggle/input/datasets
WORK_KAGGLE="$WORK_DIR/outputs"            # remplace /kaggle/working
BACKBONE_CACHE="$WORK_DIR/backbone_cache"  # remplace /kaggle/working/backbone_cache

NOTEBOOK_NAME="${NOTEBOOK%.ipynb}"
RESULTS_DIR="$WORK_DIR/results/$NOTEBOOK_NAME/exp_${EXP_NUM}"

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
#module load cudnn/8.9     2>/dev/null || true
module load python/3.11.7 2>/dev/null || true

mkdir -p "$WORK_DIR/logs"
mkdir -p "$DATA_ROOT"
mkdir -p "$WORK_KAGGLE"
mkdir -p "$BACKBONE_CACHE"
mkdir -p "$RESULTS_DIR/checkpoints"
mkdir -p "$RESULTS_DIR/history"
mkdir -p "$RESULTS_DIR/experiments"
mkdir -p "$RESULTS_DIR/saliency"

cd "$WORK_DIR"

# =============================================================================
#  Kaggle credentials (kaggle.json versionné dans le repo)
# =============================================================================
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
#
#  torch/torchvision : index officiel PyTorch CUDA 12.2
#  Tous les autres packages : versions exactes de l'environnement Kaggle
# =============================================================================
VENV_DIR="$WORK_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[SETUP] Création du virtualenv..."
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "[SETUP] Installation des dépendances..."
pip install --upgrade pip --quiet
pip uninstall -y torch torchvision

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
    scipy

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
#  Conversion notebook → script Python (tel quel)
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
#  Patch des chemins uniquement
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

# ── 1. Datasets ───────────────────────────────────────────────────────────────
for slug in [
    "samriddhibagchi/ddr-dataset-credits-to-authors/DDR-dataset",
    "samriddhibagchi/ddr-dataset-credits-to-authors",
    "aubinyoumbi/ips-node21-v1",
    "tandem03/luna16-byol-features",
    "pshikk/node-21-dataset-untampered",
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

# ── 2. backbone_cache ─────────────────────────────────────────────────────────
src = src.replace("'/kaggle/working/backbone_cache'", f"'{bc}'")
src = src.replace('"/kaggle/working/backbone_cache"', f'"{bc}"')

# ── 3. /kaggle/working ────────────────────────────────────────────────────────
src = src.replace("'/kaggle/working'", f"'{wk}'")
src = src.replace('"/kaggle/working"', f'"{wk}"')
src = re.sub(r"(['\"])/kaggle/working", lambda m: m.group(1) + wk, src)

# ── 4. Sorties finales → RESULTS_DIR ─────────────────────────────────────────
src = src.replace(f"'{wk}/checkpoints'", f"'{results_dir}/checkpoints'")
src = src.replace(f'"{wk}/checkpoints"', f'"{results_dir}/checkpoints"')

src = src.replace(f"'{wk}/results'",     f"'{results_dir}/history'")
src = src.replace(f'"{wk}/results"',     f'"{results_dir}/history"')

src = re.sub(
    r"log_experiment\(dir_path\s*=\s*['\"][^'\"]*['\"]",
    f"log_experiment(dir_path='{results_dir}/experiments'",
    src)

src = src.replace(f"'{wk}/saliency'",    f"'{results_dir}/saliency'")
src = src.replace(f'"{wk}/saliency"',    f'"{results_dir}/saliency"')
src = re.sub(
    r"save_dir\s*=\s*['\"][^'\"]*saliency['\"]",
    f"save_dir='{results_dir}/saliency'",
    src)

# ── 5. tqdm.notebook → tqdm ──────────────────────────────────────────────────
src = src.replace("from tqdm.notebook import tqdm", "from tqdm import tqdm")

# ── 6. plt.show() → plt.close('all') ─────────────────────────────────────────
src = re.sub(r'\bplt\.show\(\)', "plt.close('all')", src)

with open(script, "w") as f:
    f.write(src)

print(f"  /kaggle/input/datasets  → {dr}/")
print(f"  /kaggle/working         → {wk}")
print(f"  checkpoints             → {results_dir}/checkpoints/")
print(f"  history                 → {results_dir}/history/")
print(f"  experiments             → {results_dir}/experiments/")
print(f"  saliency                → {results_dir}/saliency/")
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

    echo ""
    echo "  Checkpoints :"
    find "$RESULTS_DIR/checkpoints" -name "*.pt" 2>/dev/null | sort \
        | while read f; do
            printf "    %-50s  %s MB\n" "$(basename "$f")" "$(du -m "$f" | cut -f1)"
          done

    echo ""
    echo "  History :"
    find "$RESULTS_DIR/history" -name "*.json" 2>/dev/null | sort \
        | while read f; do echo "    $(basename "$f")"; done

    echo ""
    echo "  Experiments :"
    find "$RESULTS_DIR/experiments" -name "*.csv" 2>/dev/null | sort \
        | while read f; do echo "    $(basename "$f")"; done

    echo ""
    echo "  Saliency :"
    PNG=$(find "$RESULTS_DIR/saliency" -name "*.png" 2>/dev/null | wc -l)
    NPY=$(find "$RESULTS_DIR/saliency" -name "*.npy" 2>/dev/null | wc -l)
    CSV=$(find "$RESULTS_DIR/saliency" -name "*.csv" 2>/dev/null | wc -l)
    echo "    $PNG figures PNG"
    echo "    $NPY fichiers NPY (patches)"
    echo "    $CSV fichiers CSV"
else
    echo ""
    echo "[ERROR] Échec — exit code $EXIT_CODE"
    echo "Log : $WORK_DIR/logs/${SLURM_JOB_NAME:-ips}_${SLURM_JOB_ID:-0}.err"
fi

deactivate
exit $EXIT_CODE
