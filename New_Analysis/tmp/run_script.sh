#!/bin/bash

echo "--- Start -----"

INPUT_PATH="/lstore/auger/work_dirs/mmartins/spacetime_clustering"

if [ -d $INPUT_PATH ]; then echo "Path exists!"; fi

FILELIST=`find $INPUT_PATH/OUTPUT_* -maxdepth 1 -type f -name "Estimators_**.parquet"`

echo $FILELIST
