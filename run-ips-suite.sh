#!/bin/bash
# =============================================================================
#  run-ips-suite.sh — Script HPC, suite IPS corrigée sur DDR et RSNA
#
#  Prérequis :
#    - Ce script, le notebook et kaggle.json dans la branche kaggle-notebooks
#
#  Datasets utilisés :
#    - samriddhibagchi/ddr-dataset-credits-to-authors   (DDR, 896)
#    - iamtapendu/rsna-pneumonia-processed-dataset      (RSNA, 1024)
#
#  Quatre modèles, 21 configurations : baseline pleine image, IPS mono-échelle,
#  MS-IPS multi-échelle à plat, H-MS-IPS en cascade à 2 ou 3 étages.
#
#  UN MODÈLE PAR JOB — le nom du run est passé en argument :
#    sbatch -J ips_ddr_ips      run-ips-suite.sh ddr_ips
#    sbatch -J ips_ddr_hmsips3  run-ips-suite.sh ddr_hmsips3
#    IPS_EPOCHS=1 sbatch run-ips-suite.sh ddr_ips        (essai court)
#
#  Sans argument, les 21 configurations tournent séquentiellement dans un
#  seul job.
#
#  Pour les lancer en parallèle, un job par modèle :
#    for r in ddr_ips ddr_msips2 ddr_hmsips3; do
#        sbatch -J ips_$r run-ips-suite.sh $r
#    done
#
#  Le notebook lit ses chemins dans des variables d'environnement, avec les
#  chemins Kaggle en valeurs par défaut. Le patch ci-dessous ne touche donc
#  qu'aux magics Jupyter, plus un filet de sécurité sur /kaggle/working.
#
#  Sorties centralisées dans :
#    ~/ips-project/results/ips-suite/
#      ├── checkpoints/   → <run_name>_weights.pt
#      ├── history/       → <run_name>_history.json, codecarbon/<run_name>/
#      ├── experiments/   → <run_name>_experiment.csv, <run_name>_selection.json
#      └── saliency/      → ddr/ et rsna/ : PNG + CSV
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

NOTEBOOK="ips-suite.ipynb"

DATASETS=(
    "samriddhibagchi/ddr-dataset-credits-to-authors"
    "iamtapendu/rsna-pneumonia-processed-dataset"
)

RUN_NAME="${1:-}"          # vide = les 21 configurations ; sinon un nom
                           # ou une liste separee par des virgules
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
echo "Run      : $TAG"
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
mkdir -p "$RESULTS_DIR/saliency/ddr"
mkdir -p "$RESULTS_DIR/saliency/rsna"

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
SCRIPT_PY="_run_${NOTEBOOK_NAME}_${TAG}_${SLURM_JOB_ID:-local}.py"

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
export IPS_RUNS="$RUN_NAME"
export IPS_EPOCHS="${IPS_EPOCHS:-}"

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
    echo "  History :"
    find "$RD/history" -maxdepth 1 -name "*_history.json" -newer "$STAMP" \
        2>/dev/null | sort | while read f; do echo "    $(basename "$f")"; done

    echo ""
    echo "  Experiments :"
    find "$RD/experiments" -type f -newer "$STAMP" 2>/dev/null | sort \
        | while read f; do echo "    $(basename "$f")"; done

    echo ""
    echo "  Saillance :"
    for DS in ddr rsna; do
        PNG=$(find "$RD/saliency/$DS" -name "*.png" -newer "$STAMP" 2>/dev/null | wc -l)
        CSV=$(find "$RD/saliency/$DS" -name "*.csv" -newer "$STAMP" 2>/dev/null | wc -l)
        if [ "$PNG" -gt 0 ] || [ "$CSV" -gt 0 ]; then
            echo "    $DS : $PNG PNG, $CSV CSV"
        fi
    done

    echo ""
    echo "── Résumé des métriques ──"
    find "$RD/experiments" -name "*_experiment.csv" -newer "$STAMP" 2>/dev/null \
        | sort | while read f; do
            echo ""
            echo "  $(basename "$f")"
            head -3 "$f" | tail -2 | cut -c1-240
          done

else
    echo ""
    echo "[ERROR] Échec — exit code $EXIT_CODE"
    echo "Log : $WORK_DIR/logs/${SLURM_JOB_NAME:-ips_suite}_${SLURM_JOB_ID:-0}.err"
fi

rm -f "$STAMP"
deactivate
exit $EXIT_CODE
