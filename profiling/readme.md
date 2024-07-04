# How to
## Build the project
```bash
mkdir build
cd build
cmake .. # on release mode 
make -j
```

## Set the number of repetitions
```bash
reps=10
gpu=rtx2000
```

## Profile the SYCL implementation with NCU (NVIDIA)
```bash
bash ../profiling/unified_profile_to_csv.sh 1 $reps $gpu
```
 

## Profile the CUDA implementation with NCU (NVIDIA)
```bash
bash ../profiling/unified_profile_to_csv.sh 2 $reps $gpu
```
 

## Profile the SYCL implementation with Vtune (Intel)
```bash
bash ../profiling/unified_profile_to_csv.sh 3 $reps $gpu
```
 

## Prepare the CSV files and generate PKL files
```bash
bash ../profiling/unified_profile_to_csv.sh 4 $reps $gpu
```
 

## Generate the overall and the detailed plots from the PKL files
- The overall plot uses all the PKL files from all the GPUs and their repetitions.
- The detailed plot uses the PKL files from all available GPUs and their 1st repetition only.

```bash
bash ../profiling/unified_profile_to_csv.sh 5 $reps $gpu
```
 