#!/bin/bash
#SBATCH
#! Number of required nodes
#SBATCH -N 1
#! tasks requested
#SBATCH -n 1
#!How much wallclock time will be required (HH:MM:SS)
#SBATCH --time=48:00:00

module load python3
source ../venv/bin/activate
python3 sphere/sphere.py