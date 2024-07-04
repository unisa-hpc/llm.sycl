#!/bin/bash

# Function to get all file of format prf_*.*.*.rep*.pkl
get_files() {
    # Directory to search in
    local dir="."

    # Initialize an empty string to hold the filenames
    local files=()

    # Loop over all files in the directory
    for file in "$dir"/prf_*.*.*.rep*.pkl
    do
        local string1=$(echo "$file" | sed -n -e 's/^.*prf_\([^\.]*\).*$/\1/p')
        local string2=$(echo "$file" | sed -n -e 's/^.*prf_[^\.]*\.\([^\.]*\).*$/\1/p')
        local string3=$(echo "$file" | sed -n -e 's/^.*\.\(.*\)\.rep.*$/\1/p')
        local integer=$(echo "$file" | sed -n -e 's/^.*rep\([0-9]*\)\.pkl$/\1/p')

        # Check if the string and integer parts are not empty and the integer part is a valid number
        if [[ -n "$string1" && -n "$string2" && -n "$string3" && -n "$integer" && $integer =~ ^[0-9]+$ ]]
        then
            # Append the filename to the files string
            files+=("${file#$dir/} ")
        fi
    done

    # Return the filenames
    echo "${files[@]}"
}

get_gpu_names_from_files() {
    # Directory to search in
    local dir="."

    # Initialize an empty string to hold the filenames
    local gpus=()

    # Loop over all files in the directory
    for file in "$dir"/prf_*.*.*.rep*.pkl
    do
        local string1=$(echo "$file" | sed -n -e 's/^.*prf_\([^\.]*\).*$/\1/p')
        local string2=$(echo "$file" | sed -n -e 's/^.*prf_[^\.]*\.\([^\.]*\).*$/\1/p')
        local string3=$(echo "$file" | sed -n -e 's/^.*\.\(.*\)\.rep.*$/\1/p')
        local integer=$(echo "$file" | sed -n -e 's/^.*rep\([0-9]*\)\.pkl$/\1/p')

        # Check if the string and integer parts are not empty and the integer part is a valid number
        if [[ -n "$string1" && -n "$string2" && -n "$string3" && -n "$integer" && $integer =~ ^[0-9]+$ ]]
        then
            # Append the filename to the files string
            gpus+=("$string3")
        fi
    done

    # removed repeated gpus from the list.
    declare -A seen
    for i in "${gpus[@]}"; do
        seen["$i"]=1
    done
    gpu_non_repeated=$(IFS=" "; echo "${!seen[*]}")

    # Return the filenames
    echo "${gpu_non_repeated[@]}"
}

job1() {
    echo "Job 1 (NCU-SYCL-LLM_SYCL) is starting..."
    for i in $(seq 1 "$reps")
    do
        echo -e "\tRepetition ${i} is running..."
        ncu --csv --set detailed --page details ./LLM_SYCL -b 1 -g 10 -x -y -s > "prf_ncu_sycl.llmsycl.${gpu_name}.rep${i}.csv"
    done
    echo "Job 1 (NCU-SYCL-LLM_SYCL) is done."
}

job2() {
    echo "Job 2 (NCU-CUDA) is starting..."
    for i in $(seq 1 "$reps")
    do
        echo -e "\tRepetition ${i} is running..."
        ncu --csv --set detailed --page details ./OrigTrain -b 1 > "prf_ncu_cuda.gold.${gpu_name}.rep${i}.csv"
    done
    echo "Job 2 (NCU-CUDA) is done."
}

job3() {
    echo "Job 3 (VTune-SYCL) is starting..."
    for i in $(seq 1 "$reps")
    do
        echo -e "\tRepetition ${i} is running..."
        vtune -collect gpu-hotspots -r "rep$i" -- ./LLM_SYCL -b 1 -g 10 -x -y -s
        vtune -report hotspots -group-by=computing-instance -format=csv -r "rep$i" > "prf_vtune_sycl.llmsycl.${gpu_name}.rep$i.csv"
    done
}

job4() {
    echo "Job 4 is running..."
    for i in $(seq 1 "$reps")
    do
        echo -e "\tRepetition ${i} is running..."
        python "${script_dir}/prepare_csv.py" --ncu "prf_ncu_sycl.llmsycl.${gpu_name}.rep${i}.csv" --type "ncu_sycl" --out "prf_ncu_sycl.llmsycl.${gpu_name}.rep${i}.pkl" --gpu "$gpu_name"
        python "${script_dir}/prepare_csv.py" --ncu "prf_ncu_cuda.gold.${gpu_name}.rep${i}.csv" --type "ncu_cuda" --out "prf_ncu_cuda.gold.${gpu_name}.rep${i}.pkl" --gpu "$gpu_name"
        python "${script_dir}/prepare_csv.py" --vtune "prf_vtune_sycl.llmsycl.${gpu_name}.rep$i.csv" --type "vtune_sycl" --out "prf_vtune_sycl.llmsycl.${gpu_name}.rep$i.pkl" --gpu "$gpu_name"
    done
    echo "Job 4 is done."
}

job5() {
    echo "Job 5 is running..."
    files=($(get_files))
    gpus=($(get_gpu_names_from_files))
    echo "Found these files (for any gpu, for any reps): ${files[*]}"
    echo "Found these gpus: ${gpus[*]}"

    # All the reps are fed to the plot.py script to generate:
    # 1. An overall device time plot for all gpus and CUDA/SYCLs for LA and NonLA.
    python "${script_dir}/plot.py" --pickle "${files[@]}"

    # 2. One set of detailed profiling plots for all reps.
    for g1 in "${gpus[@]}"
    do
        python "${script_dir}/plot.py" --pickle prf_ncu_sycl.llmsycl.${g1}.rep1.pkl prf_ncu_cuda.gold.${g1}.rep1.pkl --detailed
    done

    echo "Job 5 is done."
}


script_dir=$(dirname "$0")

# Check if a command-line argument was provided
if [ $# -ne 3 ]; then
    echo "This script should be run separately for each job on each GPU/Machine."
    echo "Need exactly three arguments <jobid> <reps> <gpu name>. For JobId, please enter a number between 1 and 5."
    echo "   1: Profile with NCU-SYCL (GPU Specific)"
    echo "   2: Profile with NCU-CUDA (GPU Specific)"
    echo "   3: Profile with VTune-SYCL (GPU Specific)"
    echo "   4: Convert the CSV files into PKL files (GPU Specific)"
    echo "   5: Generate plots from the PKL files (for all available PKL files from all the GPUs)"
    echo "The script's parent path: $script_dir"
    exit 1
fi

# Get the number from the command-line argument
num=$1
reps=$2
gpu_name=$3

# Check if the number of repetitions is a valid number
if ! [[ $reps =~ ^[0-9]+$ ]]; then
    echo "Invalid number of repetitions. Please enter a valid number."
    exit 1
fi

# Call the corresponding function based on the user's input
case $num in
    1) job1 ;;
    2) job2 ;;
    3) job3 ;;
    4) job4 ;;
    5) job5 ;;
    *) echo "Invalid input! Please enter a number between 1 and 5." ;;
esac
