#!/bin/bash

# Venv setup
set VENV="yolov8_venv"
set PROJ_NAME="mtb_cv"
set CUDA="11.8"
set GPU="num=1:j_exclusive=yes:gmodel=A30"
set PY_PATH="/home/$PROJ_NAME/virtualenvs/$VENV/bin/python"

# Load LSF and CUDA
module load lsf
module load cuda/$CUDA

set FILE_PATH="train_yolov5.py"

# GPU
bsub -Is -q gpu -gpu $GPU -R "osrel==70 && ui==aiml-python" -n 20 -P $PROJ_NAME $PY_PATH $FILE_PATH
