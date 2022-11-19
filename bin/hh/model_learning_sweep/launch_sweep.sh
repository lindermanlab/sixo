#!/bin/bash
mkdir -p reports
project_name=model_learning_sweep
seed=0

# FIVO, bootstrap proposal
# 10 runs
for i_ext in "-0.9" "-0.2" "0.5" "1.2" "1.9"
do
  for model_lr in "1e-2"
  do
    # don't decay
    sbatch --job-name=${project_name}-fivo-bs-nodecay-${i_ext} run.sh \
      ${project_name} fivo ${i_ext} ${seed} ${model_lr} "0" 1 bootstrap 0 none 0 "0" 1

    # decay 150k
    sbatch --job-name=${project_name}-fivo-bs-decay-${i_ext} run.sh \
      ${project_name} fivo ${i_ext} ${seed} ${model_lr} "150000,300000,600000" 0.5 bootstrap 0 none 0 "0" 1
  done
done

# FIVO, filtering proposal
# 40 runs
for i_ext in "-0.9" "-0.2" "0.5" "1.2" "1.9"
do
  for model_lr in "1e-3" "3e-4"
  do
    for prop_lr in "2e-4"
    do
      # don't decay
      sbatch --job-name=${project_name}-fivo-filt-nodecay-${i_ext} run.sh \
        ${project_name} fivo ${i_ext} ${seed} ${model_lr} "0" 1 filtering ${prop_lr} none 0 "0" 1

      # decay 100k
      sbatch --job-name=${project_name}-fivo-filt-decay-1-${i_ext} run.sh \
        ${project_name} fivo ${i_ext} ${seed} ${model_lr} "100000,200000,400000" 0.5 filtering ${prop_lr} none 0 "0" 1


      # decay 250k
      sbatch --job-name=${project_name}-fivo-filt-decay-2-${i_ext} run.sh \
        ${project_name} fivo ${i_ext} ${seed} ${model_lr} "250000,500000,1000000" 0.5 filtering ${prop_lr} none 0 "0" 1

      # decay 500k
      sbatch --job-name=${project_name}-fivo-filt-decay-3-${i_ext} run.sh \
        ${project_name} fivo ${i_ext} ${seed} ${model_lr} "500000,1000000,2000000" 0.5 filtering ${prop_lr} none 0 "0" 1

    done
  done
done

# SIXO, bootstrap proposal
# 90 runs
for i_ext in "-0.9" "-0.2" "0.5" "1.2" "1.9"
do
  #model_lr=5e-3
  for tilt_lr in "5e-4" "2e-4" "1e-4"
  do
    # don't decay
    sbatch --job-name=${project_name}-sixo-bs-nodecay-${i_ext} run.sh \
      ${project_name} sixo ${i_ext} ${seed} 5e-3 "0" 1 bootstrap 0 bwd_dre ${tilt_lr} "0" 1

    # decay 500k
    sbatch --job-name=${project_name}-sixo-bs-decay-${i_ext} run.sh \
      ${project_name} sixo ${i_ext} ${seed} 5e-3 "400000,800000,1600000" 0.5 bootstrap 0 bwd_dre ${tilt_lr} "100000,200000,400000" 0.5

    # decay 1M
    sbatch --job-name=${project_name}-sixo-bs-decay-slow-${i_ext} run.sh \
      ${project_name} sixo ${i_ext} ${seed} 5e-3 "800000,1600000,3200000" 0.5 bootstrap 0 bwd_dre ${tilt_lr} "200000,400000,800000" 0.5
  done

  #model_lr=1e-2
  for tilt_lr in "1e-3" "5e-4" "1e-4"
  do
    # don't decay
    sbatch --job-name=${project_name}-sixo-bs-nodecay-${i_ext} run.sh \
      ${project_name} sixo ${i_ext} ${seed} 1e-2 "0" 1 bootstrap 0 bwd_dre ${tilt_lr} "0" 1

    # decay 500k
    sbatch --job-name=${project_name}-sixo-bs-decay-${i_ext} run.sh \
      ${project_name} sixo ${i_ext} ${seed} 1e-2 "400000,800000,1600000" 0.5 bootstrap 0 bwd_dre ${tilt_lr} "100000,200000,400000" 0.5

    # decay 1M
    sbatch --job-name=${project_name}-sixo-bs-decay-slow-${i_ext} run.sh \
      ${project_name} sixo ${i_ext} ${seed} 1e-2 "800000,1600000,3200000" 0.5 bootstrap 0 bwd_dre ${tilt_lr} "200000,400000,800000" 0.5
  done
done

# SIXO, smoothing proposal
# 140 runs
for i_ext in "-0.9" "-0.2" "0.5" "1.2" "1.9"
do
  for prop_lr in "1e-4" "2e-4"
  do
    # model_lr = 3e-4
    for tilt_lr in "2e-3" "3e-3" "4e-3"
    do
      # don't decay
      sbatch --job-name=${project_name}-sixo-smooth-nodecay-${i_ext} run.sh \
        ${project_name} sixo ${i_ext} ${seed} 3e-4 "0" 1 smoothing ${prop_lr} bwd_dre ${tilt_lr} "0" 1
      # decay 100k
      sbatch --job-name=${project_name}-sixo-smooth-decay-${i_ext} run.sh \
        ${project_name} sixo ${i_ext} ${seed} 3e-4 "80000,160000,320000" 0.5 smoothing ${prop_lr} bwd_dre ${tilt_lr} "20000,40000,80000" 0.5
    done
    # model_lr = 1e-3
    for tilt_lr in "1e-3" "1.5e-3"
    do
      # don't decay
      sbatch --job-name=${project_name}-sixo-smooth-nodecay-${i_ext} run.sh \
        ${project_name} sixo ${i_ext} ${seed} 1e-3 "0" 1 smoothing ${prop_lr} bwd_dre ${tilt_lr} "0" 1
      # decay 100k
      sbatch --job-name=${project_name}-sixo-smooth-decay-${i_ext} run.sh \
        ${project_name} sixo ${i_ext} ${seed} 1e-3 "80000,160000,320000" 0.5 smoothing ${prop_lr} bwd_dre ${tilt_lr} "20000,40000,80000" 0.5
    done
    # model_lr = 3e-3
    for tilt_lr in "2e-3" "3e-3"
    do
      # don't decay
      sbatch --job-name=${project_name}-sixo-smooth-nodecay-${i_ext} run.sh \
        ${project_name} sixo ${i_ext} ${seed} 3e-3 "0" 1 smoothing ${prop_lr} bwd_dre ${tilt_lr} "0" 1
      # decay 100k
      sbatch --job-name=${project_name}-sixo-smooth-decay-${i_ext} run.sh \
        ${project_name} sixo ${i_ext} ${seed} 3e-3 "80000,160000,320000" 0.5 smoothing ${prop_lr} bwd_dre ${tilt_lr} "20000,40000,80000" 0.5
    done
  done
done
