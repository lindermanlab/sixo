#!/bin/bash
mkdir -p reports

project_name=inf_swp
seed=0
slurm_args="--job-name=${project_name}
            --time=120:00:00
            --ntasks=1
            --cpus-per-task=3
            --mem-per-cpu=3G
            --output=./reports/hh_%A_%a.out
            --error=./reports/hh_%A_%a.err"

for noise in "1." "2.5" "5." "10." "25."
do
  for subsample in "1" "2" "5" "10" "25" "50" "100"
  do
    for tilt_lr in "1e-3" "3e-3" "1e-2"
    do
      #SIXO, smoothing proposal, 42 runs
      for prop_lr in "3e-4" "1e-3"
      do
        sbatch ${slurm_args} run.sh \
          ${project_name} sixo ${seed} ${subsample} "32,32" "32" smoothing ${prop_lr} "80000,160000,320000" 0.5 bwd_dre ${tilt_lr} "20000,40000,80000" 0.5 ${noise}
      done
    done

    #SIXO, bootstrap proposal, 14 runs
    for tilt_lr in "1e-3" "5e-4"
    do
      sbatch ${slurm_args} run.sh \
        ${project_name} sixo ${seed} ${subsample} "32,32" "32" bootstrap 0 "0" 1 bwd_dre ${tilt_lr} "200000,400000,800000" 0.5 ${noise}
    done

    # FIVO, filtering proposal, 7 runs
    sbatch ${slurm_args} run.sh \
       ${project_name} fivo ${seed} ${subsample} "32,32" "32" filtering "2e-4" "100000,200000,400000" 0.5 none 0 "0" 1 ${noise}

    # FIVO, bootstrap proposal 7 runs
    sbatch ${slurm_args} run.sh \
     ${project_name} fivo ${seed} ${subsample} "0" "0" bootstrap "0" "0" 1 none 0 "0" 1 ${noise}
  done
done

