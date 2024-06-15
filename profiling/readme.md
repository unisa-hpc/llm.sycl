# How to
## Build the project
```bash
mkdir build
cd build
cmake .. # on release mode 
make -j
```

## Profile the SYCL implementation
```bash
../profiling/ncu.profile.to.csv.sh 1
```

## Profile the CUDA implementation
```bash
../profiling/ncu.profile.to.csv.sh 2
```

## Run the python script to parse and mine the csv files
```bash
python3 ../profiling/process_csv2.py
```
The generated plots will be at `../ploting/dumps/`.