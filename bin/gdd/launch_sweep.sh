#!/bin/bash

PREFIX=diffusion
mkdir -p reports

# FIVO-u
sbatch --job-name=${PREFIX}-fivo-u run.sh diffusion fivo score_fn_rb none 1 1 0 0
# FIVO-b
sbatch --job-name=${PREFIX}-fivo-b run.sh diffusion fivo none none 1 1 0 0
# SIXO-u
sbatch --job-name=${PREFIX}-sixo-u run.sh diffusion sixo score_fn_rb unified 1 1 1 0
# SIXO-b
sbatch --job-name=${PREFIX}-sixo-b run.sh diffusion sixo none unified 1 1 1 0
# SIXO-DRE
sbatch --job-name=${PREFIX}-sixo-dre run.sh diffusion sixo none dre 1 1 1 0
# SIXO-a
sbatch --job-name=${PREFIX}-sixo-a run.sh diffusion sixo none none 1 1 0 0
