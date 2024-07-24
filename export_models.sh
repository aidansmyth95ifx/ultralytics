#!/bin/bash

# Venv setup
set VENV="yolov8_p2_venv"
set PROJ_NAME="mtb_cv"
set PY_PATH="/home/$PROJ_NAME/virtualenvs/$VENV/bin/python"

# Load LSF and CUDA
module load lsf

set FILE_PATH="export_models.py"

bsub -Is -q batch -R "osrel==70 && ui==aiml_batch_training" -n 1 -P $PROJ_NAME $PY_PATH $FILE_PATH
