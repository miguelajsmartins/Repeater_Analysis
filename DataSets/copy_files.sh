#!/usr/bin/env bash

OUTPUTDIR="Vertical/MockData_Repeaters/Repeater_FixedPosAndDate_large_stats"
HOSTNAME="miguel.martins@nodo014"
PATH2FILE="/data9/mmartins/repeater_analysis_output/Vertical/Repeater_large_stats/"
FILE2COPY="OUTPUT_3834357_*/ExpRepeater_Date_2015-01-01T00:00:00_Period_86164_TotalEvents_100000_AcceptedRepEvents_100_3834357_**.parquet"

read -p "Are you sure you want to copy $FILE2COPY (y/n): " answser

if [ $answser == "y" ]
  then
    rsync -vh ${HOSTNAME}:${PATH2FILE}/${FILE2COPY} $OUTPUTDIR
    exit 0
fi
