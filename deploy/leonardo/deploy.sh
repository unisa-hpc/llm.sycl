# load nvhpc for ncu
ml nvhpc

source setenv.sh

# get the directory path of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/../.. && cd build

# Profile SYCL LLM_SYCL
bash ../profiling/unified_profile_to_csv.sh 1 4 A100

# Profile CUDA OrigTrain
bash ../profiling/unified_profile_to_csv.sh 2 4 A100

echo "Done profiling LLM_SYCL and OrigTrain."


