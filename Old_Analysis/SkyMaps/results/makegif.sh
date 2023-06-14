#!/bin/sh

list=`ls -1 Instantaneous_exposure_2022-01-01T**:00:00.0.pdf | sort -V`
output=Instantaneous_exposure_during_1Day.gif

convert -delay 20 -loop 0 $list $output

#rm -v eposlhc_Sigma_$1**.eps

echo "Done!"
