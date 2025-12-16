#!/bin/bash
#SBATCH -J run
#SBATCH -D /users/kolodny/nsegal2/
#SBATCH --job-name=embed_proteins # Job name
#SBATCH --output=/users/kolodny/nsegal2/misc/embed_proteins_1.out # Standard output and error log
#SBATCH --error=/users/kolodny/nsegal2/misc/embed_proteins_1.err # Error log
#SBATCH --gres gpu:1
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks (processes)
#SBATCH --cpus-per-task=1 # Number of CPU cores per task
#SBATCH --mem=16gb # Job memory request
#SBATCH --partition=dlc # Partition (queue) name

# Run python script
srun --container-image=/users/kolodny/nsegal2/pytorch:24.07-py3.sqsh \
--container-mounts=/users/kolodny/nsegal2/misc/:/misc_code \
--container-workdir=/misc_code \
--no-container-entrypoint \
--gres gpu:1 \
--nodes 1 \
--ntasks 1 \
--cpus-per-task 1 \
/bin/bash -c " \
python ./embed_proteins.py"
