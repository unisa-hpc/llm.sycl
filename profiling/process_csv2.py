import copy
import re
import csv
import argparse
from dataclasses import dataclass
import pprint as pp
import matplotlib.pyplot as plt
import pandas as pd

from parse_ncu_csv import DetailedEntry, InterpretNvidiaCsv
from gather_metrics1 import gather_metrics1, plot_gathered_metrics1

"""
def convert_to_dataframes(grouped_cases: dict[str, dict[str, list[ProfiledKernel]]]):
    grouped_concatenated = {}
    for kernel_name in grouped_cases:
        print("Kernel: ", kernel_name, " has ", len(list(grouped_cases[kernel_name].keys())), " grid sizes.")
        for grid_size in grouped_cases[kernel_name]:
            cases = grouped_cases[kernel_name][grid_size]
            for case in cases:
                if kernel_name not in grouped_concatenated:
                    grouped_concatenated[kernel_name] = {}
                if grid_size not in grouped_concatenated[kernel_name]:
                    grouped_concatenated[kernel_name][grid_size] = case.to_dataframe()
                else:
                    grouped_concatenated[kernel_name][grid_size] = pd.concat(
                        [grouped_concatenated[kernel_name][grid_size], case.to_dataframe()], axis=0)
    return grouped_concatenated


def plot_common_kernels(
        uut: dict[str, dict[str, list[ProfiledKernel]]],
        gold: dict[str, dict[str, list[ProfiledKernel]]]
):
    concatenated_uut = convert_to_dataframes(uut)
    concatenated_gold = convert_to_dataframes(gold)


#    uuts = concatenated_uut[kernel_name][grid_size].mean()
#    golds = concatenated_gold[kernel_name][grid_size].mean()
#
#    df = pd.concat([uuts, golds], axis=1)  # side by side
#    df.plot(kind='bar')
#    plt.title(kernel_name + " " + grid_size)
#    plt.show()


    for kernel_name in concatenated_uut:
        if kernel_name in concatenated_gold:
            lu = list(concatenated_uut[kernel_name].keys())
            lg = list(concatenated_gold[kernel_name].keys())
            print("**UUT Keys: ", lu, "\t\t**GOLD Keys:", lg)
            if len(lu) == len(lg) and len(lg) == 1:
                print("There is only one grid size for both UUT and GOLD")
                print("Assuming they are the same kernels, but different variants.")
                uuts = concatenated_uut[kernel_name][lu[0]].mean()
                golds = concatenated_gold[kernel_name][lg[0]].mean()

                profiling_col_names = list(uuts.index)
                col_names_wo_duration = copy.copy(profiling_col_names)
                col_names_wo_duration.remove("duration_ms")

                col_names_duration = ["duration_ms"]

                fig = plt.figure(figsize=(10, 9))
                fig.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.9)
                axs2 = fig.add_subplot(121)
                axs1 = fig.add_subplot(122)
                df_wo_duration = pd.concat(
                    [uuts[col_names_wo_duration], golds[col_names_wo_duration]],
                    axis=1, keys=["SYCL", "CUDA"]
                )
                df_wo_duration.plot(kind='bar', ax=axs1)

                df_duration = pd.concat(
                    [uuts[col_names_duration], golds[col_names_duration]],
                    axis=1, keys=["SYCL", "CUDA"]
                )
                df_duration.plot(kind='bar', ax=axs2)

                fig.suptitle(kernel_name + "\n" + "Work Size Gold: " + lg[0] + "\nWork Size UUT: " + lu[0])

                fig.savefig(kernel_name + ".png")
"""

if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--uut', type=str, required=True)
    argparse.add_argument('--gold', type=str, required=True)
    args = argparse.parse_args()

    interpreter_uut = InterpretNvidiaCsv(args.uut)
    interpreter_gold = InterpretNvidiaCsv(args.gold)

    raw_uut = interpreter_uut.group_by_kernel_names()
    raw_gold = interpreter_gold.group_by_kernel_names()

    preped_uut = gather_metrics1(raw_uut)
    preped_gold = gather_metrics1(raw_gold)


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


    plot_gathered_metrics1(
        preped_gold,
        preped_uut,
        name_matching,
        work_size_matching
    )

"""    
    print("UUT kernels: ")
    for name in grouped_uut:
        print('\t', name)
    print("Gold kernels: ")
    for name in grouped_gold:
        print('\t', name)

    plot_common_kernels(grouped_uut, grouped_gold)
"""
