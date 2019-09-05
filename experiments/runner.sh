#!/bin/bash

export PYTHONPATH="`pwd`/../:$PYTHONPATH"
source activate ai3

model=siamese
log_dir=../logs/$(date "+%d_%b_%Y")
log_file=$log_dir/${model}_$(date "+%H_%M_%S").log
mkdir $log_dir

echo "Log to: $log_file"
python -u ./train_sre_siamese.py > ${log_file}
