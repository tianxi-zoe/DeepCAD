#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --account=def-cbrown
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16000M
#SBATCH --gres=gpu:2
python3 demo_train_pipeline.py