#!/bin/bash
# =============================================================================
#  run-ms-ips-3scales.sh — Script HPC, MS-IPS à TROIS échelles sur DDR
#
#  Prérequis :
#    - Ce script, le notebook et kaggle.json dans la branche kaggle-notebooks
#
#  Dataset utilisé :
#    - samriddhibagchi/ddr-dataset-credits-to-authors (DDR)
#
#  Échelles : 128 / 64 / 32 px sur image 896 (pavage exact, 1029 candidats).
#  DEUX runs successifs, les deux sont nécessaires pour conclure :
#    msips3_ddr_up_v1   patch_out = 128  comparable au run à deux échelles
#    msips3_ddr_pyr_v1  patch_out =  32  pyramide stricte
#
#  Sorties centralisées dans :
#    ~/ips-project/results/ms-ips-3scales/
#      ├── ms-ips3/ddr/msips3_ddr_up_v1/   → *.pt, *_history.json, *_experiment.csv, saliency_ddr/
#      └── ms-ips3/ddr/msips3_ddr_pyr_v1/  → idem
# =============================================================================

#SBATCH --job-name=msips3_ddr
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

NOTEBOOK="ms-ips-3scales.ipynb"

DATASETS=(
    "samriddhibagchi/ddr-dataset-credits-to-authors"
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
for RUN in msips3_ddr_up_v1 msips3_ddr_pyr_v1; do
    mkdir -p "$RESULTS_DIR/ms-ips3/ddr/$RUN/saliency_ddr"
done

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

# ── 2. Dataset DDR ───────────────────────────────────────────────────────────
for slug in [
    "samriddhibagchi/ddr-dataset-credits-to-authors/DDR-dataset",
    "samriddhibagchi/ddr-dataset-credits-to-authors",
]:
    src = src.replace(f"'/kaggle/input/datasets/{slug}'", f"'{dr}/{slug}'")
    src = src.replace(f'"/kaggle/input/datasets/{slug}"', f'"{dr}/{slug}"')

src = re.sub(r"(['\"])/kaggle/input/datasets/",
             lambda m: m.group(1) + dr + "/", src)

# ── 3. backbone_cache ────────────────────────────────────────────────────────
src = src.replace("'/kaggle/working/backbone_cache'", f"'{bc}'")
src = src.replace('"/kaggle/working/backbone_cache"', f'"{bc}"')

# ── 4. Racine des sorties des deux runs ──────────────────────────────────────
#    Le runner construit f"/kaggle/working/ms-ips3/ddr/{conf.run_name}"
src = src.replace("/kaggle/working/ms-ips3", f"{results_dir}/ms-ips3")

# ── 5. log_experiment — filet de sécurité ────────────────────────────────────
src = re.sub(r"log_experiment\(dir_path\s*=\s*['\"][^'\"]*['\"]",
             f"log_experiment(dir_path='{results_dir}/ms-ips3/ddr'", src)

# ── 6. /kaggle/working générique ─────────────────────────────────────────────
src = src.replace("'/kaggle/working'", f"'{wk}'")
src = src.replace('"/kaggle/working"', f'"{wk}"')
src = re.sub(r"(['\"])/kaggle/working", lambda m: m.group(1) + wk, src)

# ── 10. tqdm.notebook → tqdm ─────────────────────────────────────────────────
src = src.replace("from tqdm.notebook import tqdm", "from tqdm import tqdm")

# ── 11. plt.show() → plt.close('all') ────────────────────────────────────────
src = re.sub(r'\bplt\.show\(\)', "plt.close('all')", src)

with open(script, "w") as f:
    f.write(src)

print("  !pip install supprimés   (codecarbon déjà installé par bash)")
print(f"  /kaggle/input/datasets  → {dr}/")
print(f"  /kaggle/working         → {wk}")
print(f"  ms-ips3/ddr             → {results_dir}/ms-ips3/ddr/<run_name>/")
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

    for RUN in msips3_ddr_up_v1 msips3_ddr_pyr_v1; do
        DDIR="$RESULTS_DIR/ms-ips3/ddr/$RUN"
        echo ""
        echo "  [$RUN]"

        echo "    Checkpoints :"
        find "$DDIR" -maxdepth 1 -name "*.pt" 2>/dev/null | sort \
            | while read f; do
                printf "      %-50s  %s MB\n" \
                    "$(basename "$f")" "$(du -m "$f" | cut -f1)"
              done

        echo "    History / experiments :"
        find "$DDIR" -maxdepth 1 \( -name "*_history.json" -o -name "*_experiment.csv" \) \
            2>/dev/null | sort | while read f; do echo "      $(basename "$f")"; done

        echo "    Confusion matrix :"
        find "$DDIR" -maxdepth 1 -name "*_confusion_matrix.png" 2>/dev/null | sort \
            | while read f; do echo "      $(basename "$f")"; done

        SDIR="$DDIR/saliency_ddr"
        echo "    Saillance : $(find "$SDIR" -name '*.png' 2>/dev/null | wc -l) PNG, "\
             "$(find "$SDIR" -name '*.npy' 2>/dev/null | wc -l) NPY, "\
             "$(find "$SDIR" -name '*.csv' 2>/dev/null | wc -l) CSV"
    done

else
    echo ""
    echo "[ERROR] Échec — exit code $EXIT_CODE"
    echo "Log : $WORK_DIR/logs/${SLURM_JOB_NAME:-msips3}_${SLURM_JOB_ID:-0}.err"
fi

deactivate
exit $EXIT_CODE
