import copy
import re
import csv
import argparse
from dataclasses import dataclass
import pprint as pp
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class ProfiledKernel:
    name_mangled: str
    name_clean: str
    duration_ms: float
    compute_throughput: float
    memory_throughput: float
    registers_per_thread: float
    grid_size: list[int]
    block_size: list[int]

    def __init__(
            self,
            name_mangled: str,
            duration_ms: str,
            compute_throughput: str,
            memory_throughput: str,
            registers_per_thread: str,
            grid_size: str,
            block_size: str
    ):
        self.name_clean = ""
        self.name_mangled = name_mangled
        self.duration_ms = float(duration_ms)
        self.compute_throughput = float(compute_throughput)
        self.memory_throughput = float(memory_throughput)
        self.registers_per_thread = float(registers_per_thread)

        self.block_size = [int(x) for x in block_size.split(',')]
        self.grid_size = [int(x) for x in grid_size.split(',')]
        self.work_size = [self.grid_size[i] * self.block_size[i] for i in range(3)]

        match = re.search(r"llmsycl::kernels::(\w+)", self.name_mangled)
        if match:
            self.name_clean = match.group(1)
            if self.name_clean.find("Kernel") != -1:
                self.name_clean = self.name_clean.replace("Kernel", "")
        else:
            if self.name_mangled.find("cutlass") != -1:
                self.name_clean = self.name_mangled[self.name_mangled.index("<"): self.name_mangled.index(">")]
            else:
                if self.name_mangled.find("_kernel") != -1:
                    self.name_clean = self.name_mangled.split("_")[0]
                else:
                    if self.name_mangled.find("_sgemm_") != -1:
                        self.name_clean = self.name_mangled
                    else:
                        print("Could not find clean name for kernel: ", self.name_mangled)
        self.name_clean = self.name_clean.lower()

    def to_dataframe(self):
        return pd.DataFrame({
            'duration_ms': [self.duration_ms],
            'compute_throughput': [self.compute_throughput],
            'memory_throughput': [self.memory_throughput],
            'registers_per_thread': [self.registers_per_thread],
        })


class InterpretNvidiaCsv:

    def __init__(self, file_path: str):
        with open(file_path, 'r') as f:
            self.reader = csv.reader(f)
            self.rows = []
            for row in self.reader:
                self.rows.append(row)
            self.header = self.rows.pop(0)

    def resolve_index_by_header_name(self, header_name: str):
        return self.header.index(header_name)

    def group_by_kernel_names(self):
        kernel_names = {}  # kernel_name -> {grid_size: [ProfiledKernel]}
        for row in self.rows:
            obj = ProfiledKernel(
                row[self.resolve_index_by_header_name('Demangled Name')],
                row[self.resolve_index_by_header_name('Duration')],
                row[self.resolve_index_by_header_name('Compute Throughput')],
                row[self.resolve_index_by_header_name('Memory Throughput')],
                row[self.resolve_index_by_header_name('# Registers')],
                row[self.resolve_index_by_header_name('Grid Size')],
                row[self.resolve_index_by_header_name('Block Size')]
            )
            kernel_name = obj.name_clean
            kernel_ws = str(obj.work_size)
            if kernel_name == "encoder":
                print("Kernel: ", kernel_name)
            if kernel_name not in kernel_names:
                kernel_names[kernel_name] = {}
            if kernel_ws not in kernel_names[kernel_name]:
                kernel_names[kernel_name][kernel_ws] = []
            kernel_names[kernel_name][kernel_ws].append(
                copy.copy(obj)
            )
        return kernel_names


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

    """
    uuts = concatenated_uut[kernel_name][grid_size].mean()
    golds = concatenated_gold[kernel_name][grid_size].mean()

    df = pd.concat([uuts, golds], axis=1)  # side by side
    df.plot(kind='bar')
    plt.title(kernel_name + " " + grid_size)
    plt.show()
    """

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


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--uut', type=str, required=True)
    argparse.add_argument('--gold', type=str, required=True)
    args = argparse.parse_args()

    interpreter_uut = InterpretNvidiaCsv(args.uut)
    grouped_uut = interpreter_uut.group_by_kernel_names()

    interpreter_gold = InterpretNvidiaCsv(args.gold)
    grouped_gold = interpreter_gold.group_by_kernel_names()

    print("UUT kernels: ")
    for name in grouped_uut:
        print('\t', name)
    print("Gold kernels: ")
    for name in grouped_gold:
        print('\t', name)

    plot_common_kernels(grouped_uut, grouped_gold)
