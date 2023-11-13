#!/usr/bin/env bash

#complain if not enough arguments are given
arg_number=$#

if [[ $arg_number != "3" || $1 == "--help" ]]
then
  echo "Please provide (in this order)"
  echo "1: Number of flares"
  echo "2: Duration of flare in days"
  echo "3: Number of events per flare"
fi

#save important variables
ud_file="./datasets/UniformDist_100000_acceptance_th80_2010-01-01_2020-01-01.parquet"
n_flares=$1
n_events=$2
flare_duration=$3

#verifies if datefile exists
if [ ! -f $ud_file ]; then echo "File does not exist! Aborting!"; exit; fi

echo "Found $ud_file!"

#search for output file
flare_dir_name="./datasets/events_with_flares/nFlares_${n_flares}_nEventsPerFlare_${n_events}_FlareDuration_${flare_duration}"

if [ -d $flare_dir_name ]; then echo "$flare_dir_name found. Deliting all the files"; rm -vr ${flare_dir_name}/*.parquet ${flare_dir_name}/*.pdf; fi

#save name of the scripts
python_flare=contaminate_ud_with_flare_events.py
python_poisson_pvalue=compute_poisson_p_value_binned_sky.py
python_lambda_pvalue=compute_lambda_pvalue_binned_sky.py
python_postrial_pvalues=compute_postrial_pvalues_blind_search.py
python_postrial_skymap=plot_postrial_pvalues_blind_search.py

#execute script that contaminates isotropic sky with flare events
if [ ! -f $python_flare ]; then echo "Requested script does not exist. Aborting!"; exit; fi

python3.8 $python_flare $ud_file $n_flares $flare_duration $n_events

flare_file=`find ${flare_dir_name}/* -type f -not -name "*PoissonPValue*" -name "*${n_flares}**${n_events}**${flare_duration}**.parquet"`

if [ ! -f $flare_file ]; then echo "File $flare_file not found! Aborting!"; exit; fi

echo "Found $flare_file"

#compute the poisson pvalues
python3.8 $python_poisson_pvalue $flare_file

#compute Lambda pvalues
poisson_file=`find $flare_dir_name -type f -name "*${n_flares}**${n_events}**${flare_duration}**PoissonPValue*.parquet"`

python3.8 $python_lambda_pvalue $poisson_file

#computes pos trial pvalues
python3.8 $python_postrial_pvalues $poisson_file

echo "... Finding postrial file ..."
postrial_file="${flare_dir_name}/BlindSearch_PosTrial_$(basename $poisson_file)"

if [ ! -f $postrial_file ]; then echo "Pos trial file not found!"; exit; fi

echo "Found $postrial_file!"

#plot skymap of flares
python3.8 $python_postrial_skymap $postrial_file

#open skymaps
evince $flare_dir_name/*.pdf &
