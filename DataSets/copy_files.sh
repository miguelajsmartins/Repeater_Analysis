#!/usr/bin/env bash

OUTPUTDIR="Vertical/MockData_Repeaters/Repeater_RandPosAndDate_large_stats"
HOSTNAME="miguel.martins@nodo014"
PATH2FILE="/data9/mmartins/repeater_analysis_output/Vertical/Repeater_RandPosAndDate_large_stats"
FILE2COPY="OUTPUT_3843630_*/IsoBG_ExpRepeater_RandPosAndDate_Period_3600_TotalEvents_100000_AcceptedRepEvents_200_RepIntensity_5_3843630_**.parquet"
read -p "Are you sure you want to copy $FILE2COPY (y/n): " answser

if [ $answser == "y" ]
  then
    rsync -vh ${HOSTNAME}:${PATH2FILE}/${FILE2COPY} $OUTPUTDIR
    exit 0
fi
