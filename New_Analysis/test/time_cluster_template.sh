#!/bin/bash

# --------------------------------------------------------------------- #
# Select partition, or queue. The LIP users should use "lipq" partition
#SBATCH -p lipq
#
#SBATCH  --mem-per-cpu=2500MB

# Transfer input files to the execution machine
# INPUT = analysis.tar.gz

#-------

#-------
# Script to launch time clustering analysis into farm
# Miguel Martins
# Sat, 17 June
#-------

echo "... loading recent version of python ..."
module load python/3.7.2

python3 --version 

WORKDIR=`pwd`
MYJOBID=`basename $WORKDIR`
echo "MYJOBID: $MYJOBID"
echo -n "hostname: "
hostname
echo "processors:"
cat /proc/cpuinfo | grep 'model name'

# to be in work directory
cd 
cd $WORKDIR

ls -lh --color

echo "--- unpacking analysis pipeline --- "

tar zxvf analysis.tar.gz
rm -fv analysis.tar.gz

cd analysis

# compile and run
echo '... Running program ...'
SCRAMBLE_OUTPUT="output_scrambling.txt"
ESTIMATOR_OUTPUT="output_estimators.txt"

echo '... Scrambling events ....'
UD_FILE=`find ./input/ -maxdepth 1 -type f -name "UniformDist**.parquet"`

echo "found $UD_FILE"

python3 scramble_events.py ${UD_FILE} &> $SCRAMBLE_OUTPUT

echo '--- Renaming scrambled file ---'
SCRAMBLED_FILE=`find ./results/ -maxdepth 1 -type f -name "Scrambled_*"`

#mv -v $SCRAMBLED_FILE ${SCRAMBLED_FILE/.parquet/_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.parquet}

#SCRAMBLED_FILE=`find ./results/ -maxdepth 1 -type f -name "Scrambled_*"`

#echo '--- Computing estimators for scrambled sky'
#python3 compute_estimators.py $SCRAMBLED_FILE &> $ESTIMATOR_OUTPUT 

#collect output and move it to lustre
OUTPUTDIR=$WORKDIR/OUTPUT_"${SLURM_ARRAY_JOB_ID}"_"$SLURM_ARRAY_TASK_ID"
OUTPATH="/lstore/auger/work_dirs/mmartins/time_clustering"

#create path if it does not exist
if [ ! -d $OUTPATH ]; then mkdir -p $OUTPATH; fi

ls -lhR

mkdir $OUTPUTDIR
mv -v ./results/*.parquet $OUTPUTDIR
mv -v $SCRAMBLE_OUTPUT "$OUTPUTDIR"
#mv -v $ESTIMATOR_OUTPUT "$OUTPUTDIR"

echo "****"

