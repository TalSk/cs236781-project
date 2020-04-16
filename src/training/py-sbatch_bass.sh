#!/bin/bash

###
# CS236781: Deep Learning
# py-sbatch.sh
#
# This script runs python from within our conda env as a slurm batch job.
# All arguments passed to this script are passed directly to the python
# interpreter.
#

###
# Example usage:
#
# Running the prepare-submission command from main.py as a batch job
# ./py-sbatch.sh main.py prepare-submission --id 123456789
#
# Running all notebooks without preparing a submission
# ./py-sbatch.sh main.py run-nb *.ipynb
#
# Running any other python script myscript.py with arguments
# ./py-sbatch.sh myscript.py --arg1 --arg2=val2
#

###
# Parameters for sbatch
#
NUM_NODES=1
NUM_CORES=2
NUM_GPUS=1
JOB_NAME="project_ddsp_job"
MAIL_USER="amitz@campus.technion.ac.il"
MAIL_TYPE=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL

###
# Conda parameters
#
CONDA_HOME=$HOME/miniconda3
CONDA_ENV=ddsp

sbatch \
	-N $NUM_NODES \
	-c $NUM_CORES \
	-w rishon3 \
	--gres=gpu:$NUM_GPUS \
	--job-name $JOB_NAME \
	--mail-user $MAIL_USER \
	--mail-type $MAIL_TYPE \
	-o 'slurm-%N-%j.out' \
<<EOF
#!/bin/bash
echo "*** SLURM BATCH JOB '$JOB_NAME' STARTING ***"

# Setup the conda env
echo "*** Activating environment $CONDA_ENV ***"
source $CONDA_HOME/etc/profile.d/conda.sh
conda activate $CONDA_ENV

ddsp_run \
  --mode=train \
  --model_dir=/home/amitz/cs236781/project/trained/bass/ \
  --gin_file=models/solo_instrument.gin \
  --gin_file=datasets/tfrecord.gin \
  --gin_param="TFRecordProvider.file_pattern='/home/amitz/cs236781/project/Dataset/Training/bass/bass.tfrecord*'" \
  --gin_param="batch_size=16" \
  --alsologtostderr

echo "*** SLURM BATCH JOB '$JOB_NAME' DONE ***"
EOF

