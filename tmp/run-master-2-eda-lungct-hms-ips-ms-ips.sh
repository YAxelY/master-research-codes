#!/bin/bash
# =============================================================================
#  run-master-2-eda-lungct-hms-ips-ms-ips.sh
#  Lance le notebook d'EDA + extraction de features multi-échelle (coarse+fine)
#  pour LUNA16 sur le cluster HPC.
#
#  Ce notebook produit UN dataset HDF5 (coarse + fine par scan) exploitable
#  à la fois par h-ms-ips-luna16 et ms-ips-luna16.
#
#  Particularité vs les templates IPS : tout le calcul intermédiaire
#  (coords + cache memmap) transite par /kaggle/tmp — ici remappé vers
#  $WORK_DIR/tmp (scratch local, pas de quota) — jamais /kaggle/working.
#  Les livrables finaux (HDF5 + EDA) vont dans $WORK_KAGGLE puis $RESULTS_DIR.
#
#  Datasets requis :
#    - vafaeii/luna16                                (CT + seg + annotations)
#    - tandem03/luna16-byol-outputs-version1         (checkpoint BYOL epoch50)
# =============================================================================

#SBATCH --job-name=eda_hmsips_msips
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

NOTEBOOK="master-2-eda-lungct-hms-ips-ms-ips.ipynb"

DATASETS=(
    "vafaeii/luna16"
    "tandem03/luna16-byol-outputs-version1"
)

# =============================================================================
#  Chemins cluster
# =============================================================================
WORK_DIR="ips-project"
DATA_ROOT="$WORK_DIR/data/datasets"        # remplace /kaggle/input/datasets
WORK_KAGGLE="$WORK_DIR/outputs"            # remplace /kaggle/working
TMP_ROOT="$WORK_DIR/tmp"                   # remplace /kaggle/tmp  (scratch, pas de quota)
BACKBONE_CACHE="$WORK_DIR/backbone_cache"  # remplace /kaggle/working/backbone_cache

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
echo "Scratch  : $TMP_ROOT"
echo "========================================="

module purge              2>/dev/null || true
module load cuda/12.6     2>/dev/null || true
module load python/3.11.7 2>/dev/null || true

mkdir -p "$WORK_DIR/logs"
mkdir -p "$DATA_ROOT"
mkdir -p "$WORK_KAGGLE"
mkdir -p "$TMP_ROOT"
mkdir -p "$BACKBONE_CACHE"
mkdir -p "$RESULTS_DIR/features"
mkdir -p "$RESULTS_DIR/eda"

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
tmp         = "$TMP_ROOT"
bc          = "$BACKBONE_CACHE"
results_dir = "$RESULTS_DIR"

with open(script) as f:
    src = f.read()

# ── 1. Supprimer les lignes !pip install ─────────────────────────────────────
src = re.sub(r'^.*!pip install.*$', '', src, flags=re.MULTILINE)

# ── 2. Datasets ──────────────────────────────────────────────────────────────
for slug in [
    "vafaeii/luna16",
    "tandem03/luna16-byol-outputs-version1",
]:
    src = src.replace(
        f"'/kaggle/input/datasets/{slug}'", f"'{dr}/{slug}'")
    src = src.replace(
        f'"/kaggle/input/datasets/{slug}"', f'"{dr}/{slug}"')

# Filet de sécurité pour tout autre /kaggle/input/datasets
src = re.sub(
    r"(['\"])/kaggle/input/datasets/",
    lambda m: m.group(1) + dr + "/",
    src)

# ── 3. /kaggle/tmp → scratch local (AVANT /kaggle/working) ───────────────────
# Le notebook fait tout le calcul intermédiaire (coords + cache memmap) ici.
src = src.replace("'/kaggle/tmp'", f"'{tmp}'")
src = src.replace('"/kaggle/tmp"', f'"{tmp}"')
src = re.sub(r"(['\"])/kaggle/tmp", lambda m: m.group(1) + tmp, src)

# ── 4. backbone_cache ────────────────────────────────────────────────────────
src = src.replace("'/kaggle/working/backbone_cache'", f"'{bc}'")
src = src.replace('"/kaggle/working/backbone_cache"', f'"{bc}"')

# ── 5. Livrables finaux → RESULTS_DIR (AVANT /kaggle/working générique) ───────
src = src.replace("'/kaggle/working/luna16_hmsips_features'", f"'{results_dir}/features'")
src = src.replace('"/kaggle/working/luna16_hmsips_features"', f'"{results_dir}/features"')
src = src.replace("'/kaggle/working/eda_luna16_multiscale'",  f"'{results_dir}/eda'")
src = src.replace('"/kaggle/working/eda_luna16_multiscale"',  f'"{results_dir}/eda"')

# ── 6. /kaggle/working générique ─────────────────────────────────────────────
src = src.replace("'/kaggle/working'", f"'{wk}'")
src = src.replace('"/kaggle/working"', f'"{wk}"')
src = re.sub(r"(['\"])/kaggle/working", lambda m: m.group(1) + wk, src)

# ── 7. tqdm.notebook → tqdm ──────────────────────────────────────────────────
src = src.replace("from tqdm.notebook import tqdm", "from tqdm import tqdm")

# ── 8. plt.show() → plt.close('all') ─────────────────────────────────────────
src = re.sub(r'\bplt\.show\(\)', "plt.close('all')", src)

with open(script, "w") as f:
    f.write(src)

print(f"  !pip install supprimés")
print(f"  /kaggle/input/datasets  → {dr}/")
print(f"  /kaggle/tmp             → {tmp}/   (scratch coords + cache memmap)")
print(f"  /kaggle/working         → {wk}")
print(f"  features (HDF5)         → {results_dir}/features/")
print(f"  eda                     → {results_dir}/eda/")
print("  Patch OK.")
PYEOF

# =============================================================================
#  Vérification post-patch
# =============================================================================
echo ""
echo "[CHECK] Validation du script généré..."

if grep -qP '[\x00-\x08\x0b\x0c\x0e-\x1f]' "$SCRIPT_PY"; then
    echo "[ERROR] Caractères de contrôle non imprimables détectés :"
    grep -naP '[\x00-\x08\x0b\x0c\x0e-\x1f]' "$SCRIPT_PY"
    exit 1
fi

if ! python -m py_compile "$SCRIPT_PY"; then
    echo "[ERROR] Erreur de syntaxe dans le script généré."
    exit 1
fi
echo "[OK] Script valide."

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
    echo "── Livrables dans $RESULTS_DIR ──"
    echo ""
    echo "  Features HDF5 (à pousser sur Kaggle comme dataset) :"
    find "$RESULTS_DIR/features" -name "*.h5" 2>/dev/null | sort \
        | while read f; do
            printf "    %-45s  %s MB\n" "$(basename "$f")" "$(du -m "$f" | cut -f1)"
          done
    echo ""
    echo "  EDA :"
    find "$RESULTS_DIR/eda" -type f 2>/dev/null | sort \
        | while read f; do echo "    $(basename "$f")"; done
    echo ""
    echo "  → Pousser $RESULTS_DIR/features/ sur Kaggle sous le slug"
    echo "    'tandem03/luna16-hmsips-features' pour les notebooks d'entraînement."
else
    echo ""
    echo "[ERROR] Échec — exit code $EXIT_CODE"
    echo "Log : $WORK_DIR/logs/${SLURM_JOB_NAME:-eda}_${SLURM_JOB_ID:-0}.err"
fi

# Nettoyage scratch (le cache memmap peut être volumineux)
echo ""
echo "[CLEANUP] Suppression du scratch $TMP_ROOT ..."
rm -rf "$TMP_ROOT" 2>/dev/null || true

deactivate
exit $EXIT_CODE
