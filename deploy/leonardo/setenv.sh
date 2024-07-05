# Making sure this script is sourced, not executed.
if [ "$0" = "$BASH_SOURCE" ]; then
    echo "This script must be sourced, not executed."
    echo "Usage: source $BASH_SOURCE"
    exit 1
fi

export LIBRARY_PATH=/leonardo/home/userexternal/${USER}/00_local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/leonardo/home/userexternal/${USER}/00_local/lib:$LD_LIBRARY_PATH
export CMAKE_PREFIX_PATH=/leonardo/home/userexternal/${USER}/00_local:$CMAKE_PREFIX_PATH

# nvhpc does not have cuBLAS
#ml nvhpc
ml cuda

# These modules for oneAPI are old.
#ml intel-oneapi-compilers
#ml intel-oneapi-mkl

# Manually installed OneAPI with the CodePlay Nvidia plugin
source /leonardo/home/userexternal/${USER}/intel/oneapi/setvars.sh

# Load OpenBLAS on the fly.
ml spack
echo "Loading OpenBLAS with spack"
spack load /kytrjfx
export OPENBLAS_INC=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-12.2.0/openblas-0.3.24-kytrjfxkavk6fcucnujsdsfqqsjfv5t7/include
export OPENBLAS_INCLUDE=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-12.2.0/openblas-0.3.24-kytrjfxkavk6fcucnujsdsfqqsjfv5t7/include
export OPENBLAS_LIB=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-12.2.0/openblas-0.3.24-kytrjfxkavk6fcucnujsdsfqqsjfv5t7/lib
export CBLAS_file=/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-12.2.0/openblas-0.3.24-kytrjfxkavk6fcucnujsdsfqqsjfv5t7/include/cblas.h

