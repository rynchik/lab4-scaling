#!/bin/bash
#SBATCH --job-name=lab4-scaling
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=lab4-%j.out

set -euo pipefail
# Перед sbatch активируйте совместимое окружение Python/MPI.
# Раздел, account, memory и интеграцию MPI со Slurm настройте по правилам кластера.
# Нужна общая файловая система с запасом места: только матрицы A–C занимают 80 ГБ.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"
python scripts/run_suite.py --profile cluster --launcher srun --tasks-per-node 32 --allow-large
python scripts/analyze.py --input report/cluster
