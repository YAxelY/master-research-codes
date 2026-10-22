#!/bin/bash
# =============================================================================
#  run-ips-gradcam.sh — Script HPC pour le notebook IPS GradCAM hybride
#
#  Variante GradCAM-guided du notebook IPS :
#    - Phase 1 : GradCAM seul sélectionne les patches (n_warmup époques)
#    - Phase 2 : Blend GradCAM + cross-attention (transition cosinus)
#    - Phase 3 : IPS natif (cross-attention seul)
#
#  Datasets utilisés :
#    - pshikk/node-21-dataset-untampered          (NODE21)
#    - tandem03/luna16-byol-features               (LUNA16)  ← skip GradCAM (is_image=False)
#    - samriddhibagchi/ddr-dataset-credits-to-authors (DDR)
#    - iamtapendu/rsna-pneumonia-processed-dataset (RSNA)
#
#  Sorties centralisées dans :
#    ~/ips-project/results/ips-gradcam/exp_<NUM>/
#      ├── runs/ips_node21_hybrid_v1/
#      │     ├── ips_node21_hybrid_v1.pt
#      │     ├── ips_node21_hybrid_v1_history.json
#      │     ├── ips_node21_hybrid_v1_experiment.csv
#      │     ├── ips_node21_hybrid_v1_hybrid.csv
#      │     ├── ips_node21_hybrid_v1_convergence.png
#      │     ├── ips_node21_hybrid_v1_confusion_matrix.png
#      │     └── saliency/
#      ├── runs/ips_luna16_fold0/          ← IPS standard (pas de GradCAM)
#      ├── runs/ips_ddr_hybrid_v1/
#      └── runs/ips_rsna_hybrid_v1/
# =============================================================================

#SBATCH --job-name=ips_gradcam
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

NOTEBOOK="ips-gradcam.ipynb"

DATASETS=(
    "pshikk/node-21-dataset-untampered"
    "tandem03/luna16-byol-features"
    "samriddhibagchi/ddr-dataset-credits-to-authors"
    "iamtapendu/rsna-pneumonia-processed-dataset"
)

EXP_NUM="001"

# =============================================================================
#  Chemins cluster
# =============================================================================
WORK_DIR="ips-project"
DATA_ROOT="$WORK_DIR/data/datasets"
WORK_KAGGLE="$WORK_DIR/outputs"
BACKBONE_CACHE="$WORK_DIR/backbone_cache"

NOTEBOOK_NAME="${NOTEBOOK%.ipynb}"
RESULTS_DIR="$WORK_DIR/results/$NOTEBOOK_NAME/exp_${EXP_NUM}"

export TORCH_HOME="$BACKBONE_CACHE"
export HF_HOME="$BACKBONE_CACHE"
export HUGGINGFACE_HUB_CACHE="$BACKBONE_CACHE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# Sous-dossiers alignés sur les RUN_NAME du notebook GradCAM
mkdir -p "$RESULTS_DIR/runs/ips_node21_hybrid_v1/saliency"
mkdir -p "$RESULTS_DIR/runs/ips_luna16_fold0"
mkdir -p "$RESULTS_DIR/runs/ips_ddr_hybrid_v1/saliency"
mkdir -p "$RESULTS_DIR/runs/ips_rsna_hybrid_v1/saliency"

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
#  scipy ajouté pour spearmanr (métriques de convergence GradCAM/IPS)
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

# ── 1. Supprimer les lignes !pip install ─────────────────────────────────────
src = re.sub(r'^.*!pip install.*$', '', src, flags=re.MULTILINE)

# ── 2. Datasets ──────────────────────────────────────────────────────────────
for slug in [
    "samriddhibagchi/ddr-dataset-credits-to-authors/DDR-dataset",
    "samriddhibagchi/ddr-dataset-credits-to-authors",
    "tandem03/luna16-byol-features",
    "pshikk/node-21-dataset-untampered",
    "iamtapendu/rsna-pneumonia-processed-dataset",
]:
    src = src.replace(
        f"'/kaggle/input/datasets/{slug}'", f"'{dr}/{slug}'")
    src = src.replace(
        f'"/kaggle/input/datasets/{slug}"', f'"{dr}/{slug}"')

src = re.sub(
    r"(['\"])/kaggle/input/datasets/",
    lambda m: m.group(1) + dr + "/",
    src)

# ── 3. backbone_cache ────────────────────────────────────────────────────────
src = src.replace("'/kaggle/working/backbone_cache'", f"'{bc}'")
src = src.replace('"/kaggle/working/backbone_cache"', f'"{bc}"')

# ── 4. SAVE_DIR par run (noms hybrides) ──────────────────────────────────────
for run_name, kaggle_path in [
    ("ips_node21_hybrid_v1", "/kaggle/working/ips/node21_hybrid"),
    ("ips_luna16_fold0",     "/kaggle/working/ips/luna16"),
    ("ips_ddr_hybrid_v1",    "/kaggle/working/ips/ddr_hybrid"),
    ("ips_rsna_hybrid_v1",   "/kaggle/working/ips/rsna_hybrid"),
]:
    target = f"{results_dir}/runs/{run_name}"
    src = src.replace(f"'{kaggle_path}'", f"'{target}'")
    src = src.replace(f'"{kaggle_path}"', f'"{target}"')

# ── 5. log_experiment / log_hybrid_experiment par run ────────────────────────
for run_name in [
    "ips_node21_hybrid_v1", "ips_luna16_fold0",
    "ips_ddr_hybrid_v1",    "ips_rsna_hybrid_v1",
]:
    target = f"{results_dir}/runs/{run_name}"
    for method in ["log_experiment", "log_hybrid_experiment"]:
        src = re.sub(
            rf"engine\.{method}\(dir_path\s*=\s*['\"][^'\"]*{run_name}['\"]",
            lambda m, t=target, meth=method: f"engine.{meth}(dir_path='{t}'",
            src)

# Filets de sécurité
src = re.sub(
    r"log_experiment\(dir_path\s*=\s*['\"][^'\"]*['\"]",
    lambda m: f"log_experiment(dir_path='{results_dir}/runs'",
    src)
src = re.sub(
    r"log_hybrid_experiment\(dir_path\s*=\s*['\"][^'\"]*['\"]",
    lambda m: f"log_hybrid_experiment(dir_path='{results_dir}/runs'",
    src)

# ── 6. save_path confusion matrix ────────────────────────────────────────────
for run_name in [
    "ips_node21_hybrid_v1", "ips_luna16_fold0",
    "ips_ddr_hybrid_v1",    "ips_rsna_hybrid_v1",
]:
    target = f"{results_dir}/runs/{run_name}"
    src = re.sub(
        rf"save_path\s*=\s*['\"][^'\"]*{run_name}_confusion_matrix\.png['\"]",
        lambda m, t=target, r=run_name: f"save_path='{t}/{r}_confusion_matrix.png'",
        src)

# ── 7. save_path convergence plot ─────────────────────────────────────────────
for run_name in ["ips_node21_hybrid_v1", "ips_ddr_hybrid_v1", "ips_rsna_hybrid_v1"]:
    target = f"{results_dir}/runs/{run_name}"
    src = re.sub(
        rf"save_path\s*=\s*['\"][^'\"]*{run_name}_convergence\.png['\"]",
        lambda m, t=target, r=run_name: f"save_path='{t}/{r}_convergence.png'",
        src)

# ── 8. Saliency save_dir ─────────────────────────────────────────────────────
for run_name in ["ips_node21_hybrid_v1", "ips_ddr_hybrid_v1", "ips_rsna_hybrid_v1"]:
    target = f"{results_dir}/runs/{run_name}/saliency"
    src = re.sub(
        rf"save_dir\s*=\s*['\"][^'\"]*{run_name}[^'\"]*saliency['\"]",
        lambda m, t=target: f"save_dir='{t}'",
        src)

# ── 9. /kaggle/working générique ─────────────────────────────────────────────
src = src.replace("'/kaggle/working'", f"'{wk}'")
src = src.replace('"/kaggle/working"', f'"{wk}"')
src = re.sub(r"(['\"])/kaggle/working", lambda m: m.group(1) + wk, src)

# ── 10. tqdm.notebook → tqdm ─────────────────────────────────────────────────
src = src.replace("from tqdm.notebook import tqdm", "from tqdm import tqdm")

# ── 11. plt.show() → plt.close('all') ────────────────────────────────────────
src = re.sub(r'\bplt\.show\(\)', "plt.close('all')", src)

with open(script, "w") as f:
    f.write(src)

print(f"  !pip install supprimés")
print(f"  /kaggle/input/datasets  → {dr}/")
print(f"  /kaggle/working         → {wk}")
print(f"  runs/ips_node21_hybrid_v1 → {results_dir}/runs/ips_node21_hybrid_v1/")
print(f"  runs/ips_luna16_fold0     → {results_dir}/runs/ips_luna16_fold0/")
print(f"  runs/ips_ddr_hybrid_v1    → {results_dir}/runs/ips_ddr_hybrid_v1/")
print(f"  runs/ips_rsna_hybrid_v1   → {results_dir}/runs/ips_rsna_hybrid_v1/")
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
    echo "── Résultats dans $RESULTS_DIR ──"

    for RUN in ips_node21_hybrid_v1 ips_luna16_fold0 ips_ddr_hybrid_v1 ips_rsna_hybrid_v1; do
        RDIR="$RESULTS_DIR/runs/$RUN"
        echo ""
        echo "  [$RUN]"

        echo "    Checkpoints :"
        find "$RDIR" -maxdepth 1 -name "*.pt" 2>/dev/null | sort \
            | while read f; do
                printf "      %-55s  %s MB\n" \
                    "$(basename "$f")" "$(du -m "$f" | cut -f1)"
              done

        echo "    History :"
        find "$RDIR" -maxdepth 1 -name "*_history.json" 2>/dev/null | sort \
            | while read f; do echo "      $(basename "$f")"; done

        echo "    Experiments :"
        find "$RDIR" -maxdepth 1 -name "*_experiment.csv" 2>/dev/null | sort \
            | while read f; do echo "      $(basename "$f")"; done

        # Fichier spécifique GradCAM hybride
        find "$RDIR" -maxdepth 1 -name "*_hybrid.csv" 2>/dev/null | sort \
            | while read f; do echo "      $(basename "$f")  [hybrid metrics]"; done

        echo "    Figures :"
        find "$RDIR" -maxdepth 1 -name "*.png" 2>/dev/null | sort \
            | while read f; do echo "      $(basename "$f")"; done

        if [ -d "$RDIR/saliency" ]; then
            PNG=$(find "$RDIR/saliency" -name "*.png" 2>/dev/null | wc -l)
            NPY=$(find "$RDIR/saliency" -name "*.npy" 2>/dev/null | wc -l)
            CSV=$(find "$RDIR/saliency" -name "*.csv" 2>/dev/null | wc -l)
            echo "    Saliency : ${PNG} PNG  ${NPY} NPY  ${CSV} CSV"
        fi
    done

else
    echo ""
    echo "[ERROR] Échec — exit code $EXIT_CODE"
    echo "Log : $WORK_DIR/logs/${SLURM_JOB_NAME:-ips}_${SLURM_JOB_ID:-0}.err"
fi

deactivate
exit $EXIT_CODE
