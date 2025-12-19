#!/bin/bash -l

#$ -P ivc-ml

#$ -pe omp 8

#$ -m bea
#$ -l gpus=1
#$ -l gpu_memory=48G
#$ -l h_rt=12:00:00
#$ -N color_rep

# Load environment
module load miniconda
conda activate colorrep

echo "Running Initial Experiment"
python -m src.experiments.steer_model_contrast --config config/steering_config_contrast.yaml

echo "Experiment (last) completed."
