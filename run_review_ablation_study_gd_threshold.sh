#!/bin/sh
### General options
### specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J review_ablation_study_gd_treshold
### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=6GB]"
### -- set the email address 
#BSUB -u s182172@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o sggm/job_logs/gpu-%J.out
#BSUB -e sggm/job_logs/gpu-%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module unload python/2.7.13_ucs4
module load python3/3.7.7
module load cuda/10.2
module load cudnn/v7.6.5.32-prod-cuda-10.2

#ENV
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
export PYTHONIOENCODING=utf8
# Our model run
cd sggm/ && python experiment.py --experiments_config configs/review_ablation_study_gd_treshold.yml --gpus -1
# Baselines run
# cd sggm/ && python baselines/run.py --experiment_name uci_carbon --n_trials 20
