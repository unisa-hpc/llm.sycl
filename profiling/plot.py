import argparse

import matplotlib.pyplot as plt
import numpy
import pickle
import numpy as np
import xarray as xr

from helpers.gather_metrics1 import ncu_gather_metrics, plot_detailed_ncu_only


def run_plot_detailed_ncu_only(gpu_name, preped_gold, preped_uut):
    """
    This is only for plotting the detailed profiling data for Nvidia NCU profiled csv on Nvidia GPUs.
    Basically an Nvidia only CUDA vs. SYCL.
    :param gpu_name:
    :param preped_gold:
    :param preped_uut:
    :return:
    """

    def name_matching(name_gold: str, name_uut: str):
        return name_gold == name_uut

    def work_size_matching(name: str, _ws_gold: str, _ws_uut: str):
        def comma_separated_to_list(s: str):
            return list(map(int, s[1:-1].split(',')))

        ws_uut = comma_separated_to_list(_ws_uut)
        ws_gold = comma_separated_to_list(_ws_gold)
        if name == "encoder":
            return ws_gold[0] * 4 == ws_uut[0] and ws_gold[1:] == ws_uut[1:]
        if name == "layernorm":
            return True

        return ws_gold == ws_uut

    plot_detailed_ncu_only(
        gpu_name,
        preped_gold,
        preped_uut,
        name_matching,
        work_size_matching
    )


if __name__ == '__main__':
    # args
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--pickle', nargs='+', type=str, required=True)
    argparse.add_argument('--detailed', action='store_true', default=False)
    args = argparse.parse_args()

    if args.detailed:
        if len(args.pickle) != 2:
            print('Detailed mode requires exactly two pickle files.')
            exit(1)

    print('Received the following list of prepared pickle files: ', args.pickle)

    all_files = {
        'ncu_cuda': {},  # gpu name
        'ncu_sycl': {},
        'vtune_sycl': {}
    }
    for f in args.pickle:
        with open(f, 'rb') as file:
            data = pickle.load(file)
            print('Loaded data from ', f)
            k1 = 'ncu_cuda' if data.get('is_ncu_cuda') else \
                'ncu_sycl' if data.get('is_ncu_sycl') else \
                    'vtune_sycl' if data.get('is_vtune_sycl') else \
                        'unknown'

            if data['gpu'] not in all_files[k1]:
                all_files[k1][data['gpu']] = []

            if data['is_ncu_cuda'] or data['is_ncu_sycl']:
                data['grouped'] = ncu_gather_metrics(data['data'])
            all_files[k1][data['gpu']].append(data)

    if args.detailed:
        print('Detailed mode enabled. '
              'This is to plot the detailed profiling data ONLY for Nvidia NCU profiled csv and ONLY for two CSVs.'
              'The gold and uut will be ncu_cuda and ncu_sycl respectively. '
              )
    else:
        print('Plotting the total device time for all the provided devices (nvidia/intel and cuda/sycl).')

    if args.detailed:
        gpu_nvidia_cuda = list(all_files['ncu_cuda'].keys())
        gpu_nvidia_sycl = list(all_files['ncu_sycl'].keys())
        if len(gpu_nvidia_cuda) != 1 or len(gpu_nvidia_sycl) != 1:
            print('Detailed mode requires exactly one Nvidia GPU CUDA and one Nvidia GPU SYCL.')
            exit(2)
        if gpu_nvidia_cuda[0] != gpu_nvidia_sycl[0]:
            print('Detailed mode requires the same Nvidia GPU for both CUDA and SYCL.')
            # exit(3)
        run_plot_detailed_ncu_only(
            gpu_nvidia_cuda[0],
            all_files['ncu_cuda'][gpu_nvidia_cuda[0]][0]['grouped'],
            all_files['ncu_sycl'][gpu_nvidia_sycl[0]][0]['grouped']
        )
    else:
        # if its a ncu_cuda, ncu_sycl
        def get_total_device_time(data_type: str, gpu_name: str, LA_only: bool):
            """
            Extracts the LA or Non-LA only total device time per run given the type and gpu name.
            So, it supports repeated runs for a single device with the same platform.
            For example, 100 runs on an RTX3060 with ncu_cuda.

            :param data_type: ncu_sycl, ncu_cuda, vtune_sycl
            :param gpu_name:
            :param LA_only: if true, only extracts the LA entries, otherwise extracts the Non-LA entries
            :return:
            """
            all_entries_kernels = []
            all_entries_device_time = []
            for profiled_run in all_files[data_type][gpu_name]:
                gathered_metrics_ncu = profiled_run['grouped']
                accu_device_time = []
                included_kernels = []
                for kn in gathered_metrics_ncu:
                    if kn == "":
                        continue
                    cond = kn.find("gemm") == -1 if LA_only else kn.find("gemm") != -1
                    if cond:
                        print("\tThe condition for GEMM has been met: ", kn)
                        continue
                    else:
                        included_kernels.append(kn)
                    for ws in gathered_metrics_ncu[kn]:
                        for entry in gathered_metrics_ncu[kn][ws]:
                            accu_device_time.append(int(entry['Duration'].iloc[0]))
                if len(all_entries_kernels) == 0:
                    all_entries_kernels = included_kernels
                else:
                    if all_entries_kernels != included_kernels:
                        print("\tThe kernels are not the same across the runs.")
                        exit(4)
                all_entries_device_time.append(np.sum(accu_device_time))
            return all_entries_kernels, all_entries_device_time


        def convert_type_to_cuda_or_sycl(t: str):
            return "CUDA" if t == "ncu_cuda" else \
                "SYCL" if t == "ncu_sycl" else \
                    "SYCL" if t == "vtune_sycl" else \
                        "unknown"


        def convert_nested_dict_to_xarray(o_data: dict):
            # simple_t: CUDA/SYCL
            # gpu: name of the GPU
            # LA: [float] which are the total device time for reps
            # NonLA: [float] which are the total device time for reps
            # each cell of xarray is a list of floats that represent the total device time for each available repetition.
            # If there is no data for a specific type or gpu, the cell will be filled with NaNs.
            all_gpus = []
            reps = 0
            for o in o_data:
                for g in o_data[o]:
                    if g not in all_gpus:
                        all_gpus.append(g)
                    if reps == 0:
                        reps = len(o_data[o][g]['LA'])
                    else:
                        if reps != len(o_data[o][g]['LA']):
                            print("The repetitions are not the same across the runs.")
                            exit(5)

            xarray = xr.Dataset(
                {

                },
                coords={
                    "Type": ["CUDA", "SYCL"],
                    "GPU": all_gpus,
                    "Reps": np.arange(reps),
                },
            )
            d1 = [[[o_data[pp][gg]["LA"][r] if pp in o_data and gg in o_data[pp] else np.nan for r in range(reps)] for gg in all_gpus] for pp in ("CUDA", "SYCL")]
            xarray["LA"] = (("Type", "GPU", "Reps"), d1)
            xarray["NonLA"] = (("Type", "GPU", "Reps"), [[[o_data[pp][gg]["NonLA"][r] if pp in o_data and gg in o_data[pp] else np.nan for r in range(reps)] for gg in all_gpus] for pp in ("CUDA", "SYCL")])
            return xarray

        # per run (we can use these data to plot the quartiles)
        overall_data = {}

        for t in all_files:
            simple_t = convert_type_to_cuda_or_sycl(t)
            for gpu in all_files[t]:
                for LA_only in [True, False]:
                    if t != "vtune_sycl":
                        kernels, device_time = get_total_device_time(t, gpu, LA_only)
                        if simple_t not in overall_data:
                            overall_data[simple_t] = {}
                        if gpu not in overall_data[simple_t]:
                            overall_data[simple_t][gpu] = {}

                        if LA_only:
                            overall_data[simple_t][gpu]['LA_kernels'] = kernels
                            overall_data[simple_t][gpu]['LA'] = device_time
                        else:
                            overall_data[simple_t][gpu]['NonLA_kernels'] = kernels
                            overall_data[simple_t][gpu]['NonLA'] = device_time
                    else:
                        if simple_t not in overall_data:
                            overall_data[simple_t] = {}
                        if gpu not in overall_data[simple_t]:
                            overall_data[simple_t][gpu] = {}
                        t1 = [i['data'][0]*1e9 for i in all_files[t][gpu]]
                        t2 = [i['data'][1]*1e9 for i in all_files[t][gpu]]
                        overall_data[simple_t][gpu]['LA'] = t1
                        overall_data[simple_t][gpu]['NonLA'] = t2
                    print("GPU: ", gpu)
                    print("Type: ", simple_t)
                    print("LA_only: ", LA_only)
                    print("Kernels: ", kernels)
                    print("Device time: ", device_time)
                    print("=====================================")

        # convert the nested dict to xarray
        xarray = convert_nested_dict_to_xarray(overall_data)
        print(xarray.to_dataframe())

        # plot data with pandas using boxplot using last dimension as the distribution
        xarray.to_dataframe().boxplot(column=['LA', 'NonLA'], by=['Type', 'GPU'],  figsize=(12, 6))

        # barplot with quartiles
        #import seaborn as sns
        #sns.set_theme(style="whitegrid")
        #sns.barplot(x="Type", y="LA", hue="GPU", data=xarray.to_dataframe())


        plt.subplots_adjust(bottom=0.3)
        plt.show()
