#!/bin/bash

# load nvhpc for ncu
ml cuda
ml nvhpc
source setenv.sh

# get the directory path of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/../.. && cd build

# Profile SYCL LLM_SYCL
bash ../profiling/unified_profile_to_csv.sh 1 4 A100

# Profile CUDA OrigTrain
bash ../profiling/unified_profile_to_csv.sh 2 4 A100

# Covert all A100 CSVs to PKL files.
bash ../profiling/unified_profile_to_csv.sh 4 4 A100

echo "Please copy the CSV and PKL files to your local machine (same dir as the profiling files from other GPUs) and run:"
echo "bash ../profiling/unified_profile_to_csv.sh 5 4 FOOBAR"
echo "Done profiling LLM_SYCL and OrigTrain."


