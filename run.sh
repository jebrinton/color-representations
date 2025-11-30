#!/bin/bash -l

#$ -P ivc-ml

#$ -pe omp 8

#$ -m bea
#$ -l gpus=1
#$ -l gpu_memory=48G
#$ -l h_rt=10:00:00
#$ -N color_rep

# Load environment
module load miniconda
conda activate colorrep

echo "Running Initial Experiment"
python -m src.experiments.get_steering_vectors --config config/config_.yaml

echo "Experiment (last) completed."
