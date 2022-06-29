#!/usr/bin/env bash

OUTPUTDIR="./Vertical/UD_large_stats/2PointCorrFunc"
HOSTNAME="miguel.martins@mastercr1"
PATH2FILE="/lustre/Auger/miguel.martins/RepeaterAnalysis/Vertical/UD_large_stats/2PointCorrFunc"
FILE2COPY="output_2PointCorrFunc_44533_*/results/2PointCorrFunc_N_100000_44533_**.parquet"

read -p "Are you sure you want to copy $FILE2COPY (y/n): " answser

if [ $answser == "y" ]
  then
    rsync -vh ${HOSTNAME}:${PATH2FILE}/${FILE2COPY} $OUTPUTDIR
    exit 0
fi
