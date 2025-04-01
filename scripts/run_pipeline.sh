#!/usr/bin/bash
#SBATCH -o out/slurm/out

# load parameters
FUNCTION=$1
CONFIG=${2:-"configs/pipeline_config.yml"}
GPU=$3
GPU_NUM=$4

# load conda
eval "$(conda shell.bash hook)"
conda activate PRAISE_ENV

# set log level
export LOGLEVEL="INFO"

# get config name
IFS='/' read -ra NAME <<< "$CONFIG"
CFG_NAME=${NAME[-1]}

# set output path
OUT="${CONFIG}.out"

# run script
if ! command -v sbatch &> /dev/null
then
	# no slurm setup: run via nohup
	export FUNCTION CONFIG OUT
    nohup sh -c 'python -u src/praise_pipeline.py $FUNCTION $CONFIG' >> $OUT 2>&1 &
else
	# run with sbatch
	sbatch <<EOT
#!/bin/bash

#SBATCH --job-name=$CFG_NAME
#SBATCH -o $OUT
#SBATCH -p $GPU
#SBATCH --gres gpu:$GPU_NUM
#SBATCH -t 0-6:00:00
#SBATCH -d singleton
#SBATCH --mem 251G

python -u src/praise_pipeline.py $FUNCTION $CONFIG
EOT
fi
