# load nvhpc for ncu
ml nvhpc

# get the directory path of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $DIR/../..
rm -rf build
mkdir build && cd build

# configure cmake and quit if it fails
cmake -DIS_LEONARDO_A100=ON ..
if [ $? -ne 0 ]; then
    echo "cmake command failed"
    exit 1
fi

make LLM_SYCL OrigTrain -j
if [ $? -ne 0 ]; then
    echo "make command failed"
    exit 1
fi

echo "Please run this command to get an interactive session:"
echo "srun -n 1 -p boost_usr_prod --ntasks-per-node=1 --gres=gpu:1 -t 00:01:00 --pty /bin/bash"
echo "Then inside it, cd to   deploy/leonardo   and run the deploy.sh script."


