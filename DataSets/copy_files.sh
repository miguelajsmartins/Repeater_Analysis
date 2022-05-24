#!/usr/bin/env bash

OUTPUTDIR="Vertical/MockData_Repeaters/Repeater_RandPosAndDate_Catalog_AugerOpenData_stats"
HOSTNAME="miguel.martins@nodo014"
PATH2FILE="/data9/mmartins/repeater_analysis_output/Vertical/Repeater_RandPosAndDate_Catalog_AugerOpenData_stats"
FILE2COPY="OUTPUT_*/REP_VerticalEvents_with_tau_**.parquet"

read -p "Are you sure you want to copy $FILE2COPY (y/n): " answser

if [ $answser == "y" ]
  then
    rsync -vh ${HOSTNAME}:${PATH2FILE}/${FILE2COPY} $OUTPUTDIR
    exit 0
fi
