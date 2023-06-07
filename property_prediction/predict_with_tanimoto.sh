#!/bin/bash

TASK=EGFR
DATA_PATH=../datasets/data/${TASK}.csv
REPRESENTATION=fingerprints
USE_PCA=False
N_TRIAL=1
TEST_SET_SIZE=0.2
USE_RMSE_CONF=True
PRECOMPUTE_REPR=True


python predict_with_tanimoto.py \
    --path ${DATA_PATH} \
    --task ${TASK} \
    --representation ${REPRESENTATION} \
    --use_pca ${USE_PCA}  \
    --n_trials ${N_TRIAL} \
    --test_set_size ${TEST_SET_SIZE} \
    --use_rmse_conf ${USE_RMSE_CONF} \
    --precompute_repr ${PRECOMPUTE_REPR}
