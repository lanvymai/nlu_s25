#!/bin/bash

#SBATCH --job-name=truthfulqa
#SBATCH --partition=n1s8-v100-1 
#SBATCH --gres=gpu:v100:1
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH -c 4
#SBATCH --requeue
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nm4867@nyu.edu
#SBATCH --account=ds_ga_1012-2025sp  # Missing account line in your version

module purge

OVERLAY_FILE=/scratch/$USER/pytorch-env/overlay-15GB-500K.ext3
SIF_IMAGE=/scratch/$USER/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

singularity exec --nv --overlay $OVERLAY_FILE:ro --bind /scratch/nm4867/nlu_s25:/scratch/nm4867/nlu_s25 $SIF_IMAGE /bin/bash -c "
source /ext3/env.sh
cd /scratch/nm4867/nlu_s25/hw3
python truthfulqa.py facebook/opt-6.7b

"
