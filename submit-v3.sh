#!/bin/bash
# =============================================================================
#  submit-v3.sh : one SLURM job per configuration of a group, for one seed.
#
#    bash submit-v3.sh focus 42       selection of 17 November, seed 42
#    bash submit-v3.sh core 42        the whole main comparison (contains focus), seed 42
#    bash submit-v3.sh ablation 42    selection module, coarse zones trained
#    bash submit-v3.sh mstudy 42      values of M
#    bash submit-v3.sh core 43        same group, another seed
#
#  Groups, in launch order, and number of jobs: focus 12, core 23, ablation 38, mstudy 54, patch 6, base 2, ms 2, lowres 2, hires 4, lowdata 6, misc 2
#  core contains focus; the other lists leave out what core runs. A model
#  already finished, or running in another job, is skipped by the notebook.
#  Generated from the configuration list of ips-suite-v3.ipynb: if a
#  configuration is added there, add it here too.
#  A finished configuration is skipped, an interrupted one resumes: running
#  the same command twice is safe.
# =============================================================================
set -euo pipefail
GROUP="${1:?usage: bash submit-v3.sh <group> [seed]}"
SEED="${2:-42}"
case "$GROUP" in
  focus) RUNS=(ddr_base896 ddr_mil128 ddr_ips64 ddr_ips32 ddr_ms64_32 ddr_h64_32 rsna_base1024 rsna_mil128 rsna_ips256 rsna_ips128 rsna_ms256_128 rsna_h256_128) ;;
  core) RUNS=(ddr_base896 ddr_base448 ddr_mil128 ddr_ips128 ddr_ips112 ddr_ips64 ddr_ips56 ddr_ips32 ddr_ms128_64 ddr_ms64_32 ddr_h224_112 ddr_h224_56 ddr_h64_32 rsna_base1024 rsna_base512 rsna_mil128 rsna_ips256 rsna_ips128 rsna_ips64 rsna_ms256_128 rsna_ms128_64 rsna_h256_128 rsna_h256_64) ;;
  ablation) RUNS=(ddr_h224_112_tc ddr_h224_56_tc rsna_h256_128_tc rsna_h256_64_tc ddr_ips128_sa ddr_ips128_rand ddr_ips112_sa ddr_ips112_rand ddr_ips64_sa ddr_ips64_rand ddr_ips56_sa ddr_ips56_rand ddr_ips32_sa ddr_ips32_rand ddr_ms128_64_sa ddr_ms128_64_rand ddr_ms64_32_sa ddr_ms64_32_rand ddr_h224_112_sa ddr_h224_112_rand ddr_h224_56_sa ddr_h224_56_rand ddr_h64_32_sa ddr_h64_32_rand rsna_ips256_sa rsna_ips256_rand rsna_ips128_sa rsna_ips128_rand rsna_ips64_sa rsna_ips64_rand rsna_ms256_128_sa rsna_ms256_128_rand rsna_ms128_64_sa rsna_ms128_64_rand rsna_h256_128_sa rsna_h256_128_rand rsna_h256_64_sa rsna_h256_64_rand) ;;
  mstudy) RUNS=(ddr_h224_56_k8 ddr_h224_56_k2 ddr_ips128_m2 ddr_ips128_m5 ddr_ips128_m20 ddr_ips128_m40 ddr_ips64_m2 ddr_ips64_m5 ddr_ips64_m20 ddr_ips64_m40 ddr_ips32_m2 ddr_ips32_m5 ddr_ips32_m20 ddr_ips32_m40 ddr_ips32_m80 ddr_ips32_m160 ddr_ms128_64_m2 ddr_ms128_64_m5 ddr_ms128_64_m20 ddr_ms128_64_m40 ddr_ms64_32_m2 ddr_ms64_32_m5 ddr_ms64_32_m20 ddr_ms64_32_m40 ddr_h224_112_m2 ddr_h224_112_m5 ddr_h224_56_m2 ddr_h224_56_m5 ddr_h224_56_m20 ddr_h224_56_m40 rsna_ips256_m2 rsna_ips256_m5 rsna_ips128_m2 rsna_ips128_m5 rsna_ips128_m20 rsna_ips128_m40 rsna_ips64_m2 rsna_ips64_m5 rsna_ips64_m20 rsna_ips64_m40 rsna_ms256_128_m2 rsna_ms256_128_m5 rsna_ms256_128_m20 rsna_ms256_128_m40 rsna_ms128_64_m2 rsna_ms128_64_m5 rsna_ms128_64_m20 rsna_ms128_64_m40 rsna_h256_128_m2 rsna_h256_128_m5 rsna_h256_64_m2 rsna_h256_64_m5 rsna_h256_64_m20 rsna_h256_64_m40) ;;
  patch) RUNS=(ddr_ips224 ddr_ips28 ddr_h224_28 ddr_h224_28_tc rsna_ips32 rsna_h256_32) ;;
  base) RUNS=(ddr_base224 rsna_base256) ;;
  ms) RUNS=(ddr_ms128_64_rs rsna_ms128_64_rs) ;;
  lowres) RUNS=(ddr_h224lr_56_tc rsna_h256lr_64_tc) ;;
  hires) RUNS=(ddr_r1792_ips128 ddr_r1792_ips128_m20 ddr_r1792_ips112 ddr_r1792_h448_112) ;;
  lowdata) RUNS=(ddr_ips128_f25 ddr_ips64_f25 ddr_ips32_f25 ddr_h224_56_f25 rsna_ips128_f25 rsna_ips64_f25) ;;
  misc) RUNS=(ddr_ips128_lazy ddr_ips128_i16) ;;
  *) echo "unknown group: $GROUP (focus core ablation mstudy patch base ms lowres hires lowdata misc)"; exit 1 ;;
esac
echo "group $GROUP, seed $SEED: ${#RUNS[@]} jobs"
for r in "${RUNS[@]}"; do
  IPS_SEEDS="$SEED" sbatch -J "v3_${r}_s${SEED}" run-ips-suite-v3.sh "$r"
done
