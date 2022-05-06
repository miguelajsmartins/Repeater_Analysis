#!/bin/sh

for file in *.parquet
do
	if [ $file != 'Hexagons.parquet' -a $file != 'Hexagons_NoBadStations.parquet' ]
	then
		rm -v $file
	fi
done

echo 'Output files removed!'
