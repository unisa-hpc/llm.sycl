#!/bin/bash

# Define the functions for each job
job1() {
    echo "Job 1 is running..."
    ncu --csv --set detailed --page details ./LLM_SYCL -s --batch 1 -x -g 10 -y >> profiled.uut.csv
}

job2() {
    echo "Job 2 is running..."
    ncu --csv --set detailed --page details ./OrigTrain -b 1 >> profiled.gold.csv
}

job3() {
    echo "Job 3 is running..."
    # Add your code for job 3 here
}

# Check if a command-line argument was provided
if [ $# -eq 0 ]; then
    echo "No arguments provided. Please enter a number between 1 and 3."
    echo "   1: Profile the SYCL implementation (UUT)"
    echo "   2: Profile the CUDA implementation (GOLD)"
    echo "   3: Run the python script and generate the plots."
    exit 1
fi

# Get the number from the command-line argument
num=$1

# Call the corresponding function based on the user's input
case $num in
    1) job1 ;;
    2) job2 ;;
    3) job3 ;;
    *) echo "Invalid input! Please enter a number between 1 and 3." ;;
esac
