#!/bin/bash
sbatch scripts/landscape_ico.sh 0
sbatch scripts/landscape_icor.sh 0
sbatch scripts/landscape_ico3.sh 0
sbatch scripts/landscape_ico3r.sh 0

for i in {1..10}
do
  sbatch scripts/landscape_ico.sh $i
  sbatch scripts/landscape_ico.sh -$i
  sbatch scripts/landscape_icor.sh $i
  sbatch scripts/landscape_icor.sh -$i
  sbatch scripts/landscape_ico3.sh $i
  sbatch scripts/landscape_ico3.sh -$i
  sbatch scripts/landscape_ico3r.sh $i
  sbatch scripts/landscape_ico3r.sh -$i
done
