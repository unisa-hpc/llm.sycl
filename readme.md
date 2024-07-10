# LLM.SYCL

This project is basically a partial translation of [LLM.C](https://github.com/karpathy/llm.c) repository from C-CUDA to
C++ SYCL.

## Contact
Please feel free to contact [me](https://salehjg.github.io/) anytime via [GitHub](https://github.com/salehjg) or email: salehjamaligolzar [at] gmail [dot] com .

## How to

### Prepare

You need to have the oneAPI and CUDA SDKs installed. The code has been tested with the following versions:

- oneAPI: 2021.4
- CUDA: 12.2

Furthermore, you need to have `numpy`, `torch`, and `python3` installed to run the training.
The dataset will be fetched automatically.

### Train

Refer to the readme file in `data/` for training the model. This is required to run the CUDA and the SYCL
implementations.

### Build

Source the oneAPI and CUDA environment and then:

```bash
mkdir build && cd build
CC=icx CXX=icpx cmake ..
# ccmake ..
make -j
```

This will give you the `LLM_SYCL`, `OrigTrain`, and `TestAll` executables.

### Run

To run the original CUDA code with minor modifications to disable training and to perform some intermediate tensor
dumping as gold values:

```bash
./OrigTrain -b 1
```

To run the SYCL code:

```bash
./LLM_SYCL -s --batch 1 -x -g 10 -y
```

Set `-g` for larger values to generate more text. See `-h` for more details.

To run the test suite:

```bash
./TestAll
```

### Verify

The output of the SYCL code should be similar to the output of the CUDA code.
Other than that, for more detailed comparison with the gold (CUDA) implementation, you can use the `data/compare.py`
script:

```bash
./build/OrigTrain -b 1
./build/LLM_SYCL -s --batch 1 -g 10
python data/compare.py
```

Note that we are running the SYCL implementation with profiling and intermediate tensor dumping enabled.
This is the default config for the modified CUDA implementation.

## Credits

This repo is developed as the final project for the HPC course 2024 of Prof. B. Cosenza at the University of Salerno.
The following open-source projects have been used:

- [GTest](https://github.com/google/googletest) (system-wide)
- [SpdLog](https://github.com/gabime/spdlog) (system-wide)
- [oneMKL-Interface](https://github.com/oneapi-src/oneMKL) (with `FetchContent`)
- [libnpy](https://github.com/llohse/libnpy) (with `FetchContent`)
- [argparse](https://github.com/p-ranav/argparse) (with `FetchContent`)
