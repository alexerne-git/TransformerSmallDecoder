#! /bin/bash
#SBATCH --partition=shared-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64000 
#SBATCH --time=00:20:00 

module load Anaconda3

source ~/.bashrc
conda activate /home/users/e/USERNAME/.conda/envs/condaenv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Checking Python executable and version..."
which python  
python --version 

python main.py 