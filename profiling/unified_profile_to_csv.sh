#!/bin/bash

job1() {
    echo "Job 1 (NCU-SYCL-LLM_SYCL) is starting..."
    for i in $(seq 0 "$reps")
    do
        echo -e "\tRepetition ${i} is running..."
        ncu --csv --set detailed --page details ./LLM_SYCL -b 1 -g 10 -x -y -s > "prf_ncu_sycl.llmsycl.rep${i}.csv"
    done
    echo "Job 1 (NCU-SYCL-LLM_SYCL) is done."
}

job2() {
    echo "Job 2 (NCU-CUDA) is starting..."
    for i in $(seq 0 "$reps")
    do
        echo -e "\tRepetition ${i} is running..."
        ncu --csv --set detailed --page details ./OrigTrain -b 1 > "prf_ncu_cuda.gold.rep${i}.csv"
    done
    echo "Job 2 (NCU-CUDA) is done."
}

job3() {
    echo "Job 3 (VTune-SYCL) is starting..."
    for i in $(seq 0 "$reps")
    do
        echo -e "\tRepetition ${i} is running..."
        vtune -collect gpu-hotspots -r "rep$i" -- ./LLM_SYCL -b 1 -g 10 -x -y -s
        vtune -report hotspots -group-by=computing-instance -format=csv -r "rep$i" > "prf_vtune_sycl.llmsycl.rep$i.csv"
    done
}

job4() {
    echo "Job 4 is running..."
    for i in $(seq 0 "$reps")
    do
        echo -e "\tRepetition ${i} is running..."
        python prepare_csv.py --ncu "prf_ncu_sycl.llmsycl.rep${i}.csv" --type "ncu_sycl" --out "prf_ncu_sycl.llmsycl.rep${i}.pkl"
        python prepare_csv.py --ncu "prf_ncu_cuda.gold.rep${i}.csv" --type "ncu_cuda" --out "prf_ncu_cuda.gold.rep${i}.pkl"
        python prepare_csv.py --vtune "prf_vtune_sycl.llmsycl.rep$i.csv" --type "vtune_sycl" --out "prf_vtune_sycl.llmsycl.rep$i.pkl"
    done

    plots_args=""
    for i in $(seq 0 "$reps")
    do
        plots_args+="prf_ncu_sycl.llmsycl.rep${i}.pkl "
        plots_args+="prf_ncu_cuda.gold.rep${i}.pkl "
        plots_args+="prf_vtune_sycl.llmsycl.rep${i}.pkl "
    done

    # All the reps are fed to the plot.py script to generate:

    # 1. An overall device time plot for all gpus and CUDA/SYCLs for LA and NonLA.
    python plot.py --pickle "$plots_args"

    # 2. One set of detailed profiling plots for all reps.
    python plot.py --pickle "$plots_args" --detailed

    echo "Job 4 is done."
}

# Check if a command-line argument was provided
if [ $# -ne 2 ]; then
    echo "Need exactly two arguments <jobid> <reps>. For JobId, please enter a number between 1 and 4."
    echo "   1: Profile with NCU-CUDA"
    echo "   2: Profile with NCU-SYCL"
    echo "   3: Profile with VTune-SYCL"
    echo "   4: Run the python script and generate the plots."
    exit 1
fi

# Get the number from the command-line argument
num=$1
reps=$2

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
    *) echo "Invalid input! Please enter a number between 1 and 4." ;;
esac
