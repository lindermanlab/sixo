#!/bin/bash
mkdir -p reports
project_name=sixo_svm
job_prefix=${project_name}

for seed in 0 1 2 3 4
do
  for lr in "3e-4" "6e-4" "1e-4"
  do
      for tilt_lr in "1e-3" "2e-3" "5e-4"
      do
        # SIXO-DRE
        sbatch --job-name=${job_prefix}-sixo run.sh ${project_name} sixo multinomial ${lr} bwd_dre ${tilt_lr} ${seed}
      done

      # FIVO
      sbatch --job-name=${job_prefix}-fivo run.sh ${project_name} fivo multinomial ${lr} none 0.0 ${seed}

      # SIXO-q
      sbatch --job-name=${job_prefix}-sixo-q run.sh ${project_name} sixo multinomial ${lr} quadrature 0.0 ${seed}
    done

    # IWAE
    sbatch --job-name=${job_prefix}-iwae run.sh ${project_name} iwae multinomial ${lr} none 0.0 ${seed}
done
