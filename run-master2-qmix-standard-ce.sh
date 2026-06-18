#!/bin/bash
# =============================================================================
#  run-master2-qmix-standard-ce.sh — Script HPC pour le notebook QMix + Standard CE (DDR)
#
#  Prérequis :
#    - Ce script, le notebook et kaggle.json sont dans la branche kaggle-notebooks
#
#  Dataset utilisé :
#    - samriddhibagchi/ddr-dataset-credits-to-authors (DDR)
#
#  Sorties centralisées dans :
#    ~/ips-project/results/master2-qmix-standard-ce/exp_<NUM>/
#      ├── runs/ce_ddr_v1/
#      │     ├── ce_ddr_v1.pt
#      │     ├── ce_ddr_v1_weights.pt
#      │     ├── ce_ddr_v1_history.json
#      │     ├── ce_ddr_v1_experiment.csv
#      │     └── ce_ddr_v1_confusion_matrix.png
#      └── runs/qmix_ddr_v1/
#            ├── qmix_ddr_v1.pt
#            ├── qmix_ddr_v1_weights.pt
#            ├── qmix_ddr_v1_history.json
#            ├── qmix_ddr_v1_experiment.csv
#            └── qmix_ddr_v1_confusion_matrix.png
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
#  CONFIGURATION
# =============================================================================

NOTEBOOK="master2-qmix-standard-ce.ipynb"

DATASETS=(
    "samriddhibagchi/ddr-dataset-credits-to-authors"
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

# Sous-dossiers alignés sur les CKPT_DIR définis dans le notebook
mkdir -p "$RESULTS_DIR/runs/ce_ddr_v1"
mkdir -p "$RESULTS_DIR/runs/qmix_ddr_v1"

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
#  codecarbon installé ici — pas besoin de !pip install dans le notebook
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

# ── 1. Supprimer les lignes !pip install (magics Jupyter) ────────────────────
src = re.sub(r'^.*!pip install.*$', '', src, flags=re.MULTILINE)

# ── 2. Datasets ──────────────────────────────────────────────────────────────
for slug in [
    "samriddhibagchi/ddr-dataset-credits-to-authors/DDR-dataset",
    "samriddhibagchi/ddr-dataset-credits-to-authors",
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

# ── 4. CKPT_DIR_CE ───────────────────────────────────────────────────────────
src = src.replace(
    "'/kaggle/working/runs/ce_ddr_v1'",
    f"'{results_dir}/runs/ce_ddr_v1'")
src = src.replace(
    '"/kaggle/working/runs/ce_ddr_v1"',
    f'"{results_dir}/runs/ce_ddr_v1"')

# ── 5. CKPT_DIR_QMIX ─────────────────────────────────────────────────────────
src = src.replace(
    "'/kaggle/working/runs/qmix_ddr_v1'",
    f"'{results_dir}/runs/qmix_ddr_v1'")
src = src.replace(
    '"/kaggle/working/runs/qmix_ddr_v1"',
    f'"{results_dir}/runs/qmix_ddr_v1"')

# ── 6. /kaggle/working générique ─────────────────────────────────────────────
src = src.replace("'/kaggle/working'", f"'{wk}'")
src = src.replace('"/kaggle/working"', f'"{wk}"')
src = re.sub(r"(['\"])/kaggle/working", lambda m: m.group(1) + wk, src)

# ── 7. log_experiment CE ─────────────────────────────────────────────────────
# IMPORTANT : on utilise une fonction lambda comme remplacement, et NON une
# chaine contenant "\\1". re.sub interprète les backslashes dans une chaine
# de remplacement (\1, \g<1>, etc.) — combiné à un f-string, cela pouvait
# produire des octets de contrôle invisibles (ex: U+0001) dans le script
# généré, provoquant un "SyntaxError: invalid non-printable character".
# Une fonction lambda est inséré littéralement, sans aucune réinterprétation.
src = re.sub(
    r"ce_engine\.log_experiment\(CKPT_DIR_CE\)",
    lambda m: f"ce_engine.log_experiment('{results_dir}/runs/ce_ddr_v1')",
    src)

# ── 8. log_experiment QMix ───────────────────────────────────────────────────
src = re.sub(
    r"qm_engine\.log_experiment\(CKPT_DIR_QMIX\)",
    lambda m: f"qm_engine.log_experiment('{results_dir}/runs/qmix_ddr_v1')",
    src)

# Filet de sécurité log_experiment (idem : lambda, pas de backreference)
src = re.sub(
    r"log_experiment\(dir_path\s*=\s*['\"][^'\"]*['\"]\)",
    lambda m: f"log_experiment(dir_path='{results_dir}/runs')",
    src)

# ── 9. save_path confusion matrix CE ─────────────────────────────────────────
src = re.sub(
    r"save_path\s*=\s*['\"][^'\"]*ce_ddr_v1_confusion_matrix\.png['\"]",
    lambda m: f"save_path='{results_dir}/runs/ce_ddr_v1/ce_ddr_v1_confusion_matrix.png'",
    src)

# ── 10. save_path confusion matrix QMix ──────────────────────────────────────
src = re.sub(
    r"save_path\s*=\s*['\"][^'\"]*qmix_ddr_v1_confusion_matrix\.png['\"]",
    lambda m: f"save_path='{results_dir}/runs/qmix_ddr_v1/qmix_ddr_v1_confusion_matrix.png'",
    src)

# ── 11. tqdm.notebook → tqdm ─────────────────────────────────────────────────
src = src.replace("from tqdm.notebook import tqdm", "from tqdm import tqdm")

# ── 12. plt.show() → plt.close('all') ────────────────────────────────────────
src = re.sub(r'\bplt\.show\(\)', "plt.close('all')", src)

with open(script, "w") as f:
    f.write(src)

print(f"  !pip install supprimés   (codecarbon déjà installé par bash)")
print(f"  /kaggle/input/datasets  → {dr}/")
print(f"  /kaggle/working         → {wk}")
print(f"  runs/ce_ddr_v1          → {results_dir}/runs/ce_ddr_v1/")
print(f"  runs/qmix_ddr_v1        → {results_dir}/runs/qmix_ddr_v1/")
print("  Patch OK.")
PYEOF

# =============================================================================
#  Vérification post-patch : syntaxe valide + absence de caractères de contrôle
#  (ajouté pour détecter immédiatement toute corruption introduite par le
#  patch ci-dessus, plutôt que d'échouer après plusieurs heures de calcul)
# =============================================================================
echo ""
echo "[CHECK] Validation du script généré..."

if grep -qP '[\x00-\x08\x0b\x0c\x0e-\x1f]' "$SCRIPT_PY"; then
    echo "[ERROR] Caractères de contrôle non imprimables détectés dans $SCRIPT_PY :"
    grep -naP '[\x00-\x08\x0b\x0c\x0e-\x1f]' "$SCRIPT_PY"
    exit 1
fi

if ! python -m py_compile "$SCRIPT_PY"; then
    echo "[ERROR] Le script généré contient une erreur de syntaxe — voir ci-dessus."
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

    for RUN in ce_ddr_v1 qmix_ddr_v1; do
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

        echo "    Confusion matrix :"
        find "$RDIR" -maxdepth 1 -name "*_confusion_matrix.png" 2>/dev/null | sort \
            | while read f; do echo "      $(basename "$f")"; done
    done

else
    echo ""
    echo "[ERROR] Échec — exit code $EXIT_CODE"
    echo "Log : $WORK_DIR/logs/${SLURM_JOB_NAME:-ips}_${SLURM_JOB_ID:-0}.err"
fi

deactivate
exit $EXIT_CODE
