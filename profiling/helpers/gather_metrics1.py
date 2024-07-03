from helpers.parse_ncu_csv import DetailedEntry
import pandas as pd
from typing import Callable
import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pathlib
import numpy as np


def ncu_gather_metrics(parsed_csv: dict[str, dict[int, DetailedEntry]]):
    """
    Gathers metrics and stores them by {kernel_name: {work_size: [DetailedEntry]}}
    :param parsed_csv:
    :return:
    """
    gathered: dict[str: dict[str: pd.DataFrame]] = {}
    for kn in parsed_csv:
        for kid in parsed_csv[kn]:
            entry = parsed_csv[kn][kid]
            if kn not in gathered:
                gathered[kn] = {}
            if str(entry.work_size) not in gathered[kn]:
                gathered[kn][str(entry.work_size)] = []

            gathered[kn][str(entry.work_size)].append(
                pd.DataFrame({
                    'Duration': [entry.metrics['Duration']],
                    'Compute (SM) Throughput': [entry.metrics['Compute (SM) Throughput']],
                    'DRAM Throughput': [entry.metrics['DRAM Throughput']],
                    'L1/TEX Cache Throughput': [entry.metrics['L1/TEX Cache Throughput']],
                    'L2 Cache Throughput': [entry.metrics['L2 Cache Throughput']],
                    'L1/TEX Hit Rate': [entry.metrics['L1/TEX Hit Rate']],
                    'L2 Hit Rate': [entry.metrics['L2 Hit Rate']],
                    'Achieved Occupancy': [entry.metrics['Achieved Occupancy']],

                    'Avg. Divergent Branches': [entry.metrics['Avg. Divergent Branches']],
                    'Registers Per Thread': [entry.metrics['Registers Per Thread']],
                    'Static Shared Memory Per Block': [entry.metrics['Static Shared Memory Per Block']],
                    'Dynamic Shared Memory Per Block': [entry.metrics['Dynamic Shared Memory Per Block']],
                })
            )
    return gathered


def seaborn_grouped_bargraph(
        list_cuda: list[float],
        list_sycl: list[float],
        gpu_name: str,
        dump_dir="dumps/"
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Create a pandas DataFrame with your data
    df = pd.DataFrame({
        'Kernel Types': ['Non-LA', 'LA'],
        'CUDA and cuBLASLt': list_cuda,
        'SYCL and OneMKL Interface': list_sycl
    })

    # Melt the DataFrame to create separate rows for each value
    melted_df = df.melt(id_vars='Kernel Types', value_vars=['CUDA and cuBLASLt', 'SYCL and OneMKL Interface'])

    # Set seaborn plotting aesthetics
    sns.set(style='white')
    sns.set_style('whitegrid')

    # Create the grouped bar chart
    ax = sns.barplot(x='Kernel Types', y='value', hue='variable', data=melted_df, palette=['lime', 'turquoise'],
                     dodge=True)

    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10)

    # Add axis titles
    plt.xlabel('Kernel Types')
    plt.ylabel('Accumulated Device Time (ms)')
    plt.title('CUDA vs SYCL (' + gpu_name + ')')

    # Show the plot
    plt.savefig(pathlib.Path(dump_dir).joinpath("device_time_comparison.svg"))
    plt.savefig(pathlib.Path(dump_dir).joinpath("device_time_comparison.png"))


def plot_detailed_ncu_only(
        gpu_name: str,
        gathered_metrics_gold: dict[str, dict[str, pd.DataFrame]],
        gathered_metrics_uut: dict[str, dict[str, pd.DataFrame]],
        lambda_name_matching: Callable[[str, str], bool],
        lambda_work_size_matching: Callable[[str, str, str], bool],
        dump_dir="dumps/",
        overall_device_time_only=False
):
    # Overall device time comparison gold vs uut
    my_kernels_only_cuda = 0
    my_kernels_only_sycl = 0
    blas_kernels_only_cuda = 0
    blas_kernels_only_sycl = 0

    accu_device_time_gold = []
    accu_device_time_uut = []
    included_kernels = []
    for kn in gathered_metrics_gold:
        if kn.find("gemm") == -1:
            print("Has no GEMM: ", kn)
            continue
        else:
            included_kernels.append(kn)
        for ws in gathered_metrics_gold[kn]:
            for entry in gathered_metrics_gold[kn][ws]:
                accu_device_time_gold.append(int(entry['Duration'].iloc[0]))
    for kn in gathered_metrics_uut:
        if kn.find("gemm") == -1:
            print("Has no GEMM: ", kn)
            continue
        else:
            included_kernels.append(kn)
        for ws in gathered_metrics_uut[kn]:
            for entry in gathered_metrics_uut[kn][ws]:
                accu_device_time_uut.append(int(entry['Duration'].iloc[0]))

    print("=====================================================")
    print("Included kernels: ", included_kernels)
    print("Accumulated device time: ", np.sum(accu_device_time_gold))
    print("Accumulated device time: ", np.sum(accu_device_time_uut))
    my_kernels_only_cuda = np.sum(accu_device_time_gold)
    my_kernels_only_sycl = np.sum(accu_device_time_uut)

    accu_device_time_gold = []
    accu_device_time_uut = []
    included_kernels = []

    for kn in gathered_metrics_gold:
        if kn.find("gemm") != -1:
            print("Has GEMM: ", kn)
            continue
        else:
            included_kernels.append(kn)
        for ws in gathered_metrics_gold[kn]:
            for entry in gathered_metrics_gold[kn][ws]:
                accu_device_time_gold.append(int(entry['Duration'].iloc[0]))
    for kn in gathered_metrics_uut:
        if kn.find("gemm") != -1:
            print("Has GEMM: ", kn)
            continue
        else:
            included_kernels.append(kn)
        for ws in gathered_metrics_uut[kn]:
            for entry in gathered_metrics_uut[kn][ws]:
                accu_device_time_uut.append(int(entry['Duration'].iloc[0]))

    print("=====================================================")
    print("Included kernels: ", included_kernels)
    print("Accumulated device time gold: ", np.sum(accu_device_time_gold))
    print("Accumulated device time uut: ", np.sum(accu_device_time_uut))
    blas_kernels_only_cuda = np.sum(accu_device_time_gold)
    blas_kernels_only_sycl = np.sum(accu_device_time_uut)

    # Specify the values of blue bars (height)
    cuda_data = [my_kernels_only_cuda / 1000000.0, blas_kernels_only_cuda / 1000000.0]
    # Specify the values of orange bars (height)
    sycl_data = [my_kernels_only_sycl / 1000000.0, blas_kernels_only_sycl / 1000000.0]

    if overall_device_time_only:
        return {'cuda': cuda_data, 'sycl': sycl_data, 'gpu_name': gpu_name}

    seaborn_grouped_bargraph(cuda_data, sycl_data, gpu_name, dump_dir=dump_dir)

    # Per kernel comparison (only for kernels with exact same names)
    for kn in gathered_metrics_gold:
        for kn2 in gathered_metrics_uut:
            if lambda_name_matching(kn, kn2):
                print("Found a match for kernel names: ", kn, " and ", kn2)
                for ws in gathered_metrics_gold[kn]:
                    for ws2 in gathered_metrics_uut[kn2]:
                        if lambda_work_size_matching(kn, ws, ws2):
                            print("Found a match for work sizes: ", ws, " and ", ws2)

                            df_gold = pd.concat(gathered_metrics_gold[kn][ws], axis=0)
                            df_gold = df_gold.mean()

                            df_uut = pd.concat(gathered_metrics_uut[kn2][ws2], axis=0)
                            df_uut = df_uut.mean()

                            ###############################################################
                            abs_cols = ["Duration",
                                        # "Memory Throughput",
                                        "Avg. Divergent Branches",
                                        "Static Shared Memory Per Block",
                                        "Dynamic Shared Memory Per Block", ]
                            profiling_col_names = list(df_uut.index)
                            col_names_wo_absolute = copy.copy(profiling_col_names)
                            for c in abs_cols:
                                col_names_wo_absolute.remove(c)
                            col_names_absolute_only = abs_cols

                            spec = gridspec.GridSpec(ncols=2, nrows=1,
                                                     width_ratios=[4, 7], wspace=0.2,
                                                     hspace=0.5, height_ratios=[1])
                            fig = plt.figure(figsize=(10, 8))
                            fig.subplots_adjust(left=0.1, right=0.95, bottom=0.35, top=0.9)
                            axs2 = fig.add_subplot(spec[1])
                            axs1 = fig.add_subplot(spec[0])
                            df_wo_absolute = pd.concat(
                                [df_uut[col_names_wo_absolute], df_gold[col_names_wo_absolute]],
                                axis=1, keys=["SYCL", "CUDA"]
                            )
                            df_wo_absolute.plot(kind='bar', ax=axs2, color=['turquoise', 'lime'])  # , log=True)

                            df_absolute_only = pd.concat(
                                [df_uut[col_names_absolute_only], df_gold[col_names_absolute_only]],
                                axis=1, keys=["SYCL", "CUDA"]
                            )
                            df_absolute_only.plot(kind='bar', ax=axs1, color=['turquoise', 'lime'])

                            axs1.get_legend().remove()

                            axs1.bar_label(axs1.containers[-1], padding=3, rotation=90, fontsize=4)
                            axs1.bar_label(axs1.containers[0], padding=3, rotation=90, fontsize=4)
                            axs2.bar_label(axs2.containers[-1], padding=3, rotation=90, fontsize=4)
                            axs2.bar_label(axs2.containers[0], padding=3, rotation=90, fontsize=4)

                            fig.suptitle(kn + "\n" + "Work Size Gold: " + ws + "\nWork Size UUT: " + ws2)
                            fig.savefig(pathlib.Path(dump_dir).joinpath(kn + ".svg"))

                            ###############################################################
