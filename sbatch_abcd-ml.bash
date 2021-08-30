#!/bin/bash
#SBATCH -J abcdml
#SBATCH --time=2-00:00:00
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=5G
#SBATCH -p ncf
#SBATCH --account=mclaughlin_lab
# Outputs ----------------------------------
#SBATCH -o log/%x-%A_%a.out


CORES="${SLURM_CPUS_PER_TASK}"

N=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source ~/code/PYTHON_MODULES.txt
source activate abcd_ml_3.7

set -aeuxo pipefail

PA=$( awk -v N="${N}" 'FNR == N { print $1 }' arguments_file.txt )
SUM=$( awk -v N="${N}" 'FNR == N { print $2 }' arguments_file.txt )
OUTCOME=$( awk -v N="${N}" 'FNR == N { print $3 }' arguments_file.txt )
TIME=$( awk -v N="${N}" 'FNR == N { print $4 }' arguments_file.txt )

srun -c "${CORES}" python abcd-ml.py -p "${PA}" -s "${SUM}" -t "${TIME}" -y "${OUTCOME}" -c "${CORES}" -ni 5 -no 20 --slurmid "${SLURM_ARRAY_JOB_ID}_${N}"

#14,27,28,41,42