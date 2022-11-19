#!/bin/bash
mkdir -p reports

project_name=eval_sweep
seed=0

# 20 runs
for i_ext in "-0.9" "-0.2" "0.5" "1.2" "1.9"
do
  # FIVO, bootstrap proposal, decay 150k
  sbatch --job-name=eval-${project_name}-fivo-bs-${i_ext} run.sh \
    ${project_name} fivo ${i_ext} ${seed} "1e-2" "150000,300000,600000" 0.5 bootstrap 0 none 0 "0" 1

  # FIVO, filtering proposal
  sbatch --job-name=eval-${project_name}-fivo-filt-${i_ext} run.sh \
    ${project_name} fivo ${i_ext} ${seed} "3e-4" "0" 1 filtering "2e-4" none 0 "0" 1

  # SIXO, bootstrap proposal
  sbatch --job-name=eval-${project_name}-sixo-bs-${i_ext} run.sh \
    ${project_name} sixo ${i_ext} ${seed} "5e-3" "400000,800000,1600000" 0.5 bootstrap 0 bwd_dre "2e-4" "100000,200000,400000" 0.5

  # SIXO, smoothing proposal
  sbatch --job-name=eval-${project_name}-sixo-smooth-${i_ext} run.sh \
    ${project_name} sixo ${i_ext} ${seed} "3e-4" "80000,160000,320000" 0.5 smoothing "1e-4" bwd_dre "4e-3" "20000,40000,80000" 0.5
done
