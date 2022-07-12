#SBATCH
#!/bin/bash
#! Which partition (queue) should be used
#SBATCH -p cosmosx
#! Number of required nodes
#SBATCH -N 1
#!How much wallclock time will be required (HH:MM:SS)
#SBATCH --time=12:00:00

module load miniconda3
source venv/bin/activate
python examples/ico/landscape_ico.py $1
