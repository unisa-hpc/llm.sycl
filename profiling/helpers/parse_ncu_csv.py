import copy
import re
import csv
import argparse
from dataclasses import dataclass
import pprint as pp
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class DetailedEntry:
    launch_id: int = 0
    metrics: dict[str, float] = None  # metric_name -> value
    metrics_str: dict[str, str] = None  # metric_name -> value
    metrics_units: dict[str, str] = None  # metric_name -> unit
    metrics_sections: dict[str, str] = None  # metric_name -> section name
    stream: int = 0
    context: int = 0
    process_name: str = ""
    device: int = 0
    compute_capability: str = ""
    block_size = []
    grid_size = []
    work_size = []

    def __init__(
            self,
            launch_id: str,
            stream: str,
            context: str,
            process_name: str,
            device: str,
            compute_capability: str,
            block_size: str,
            grid_size: str
    ):
        self.launch_id = int(launch_id)
        self.stream = int(stream)
        self.context = int(context)
        self.process_name = process_name
        self.device = int(device)
        self.compute_capability = compute_capability
        self.block_size = [int(x) for x in block_size[1:-1].split(',')]
        self.grid_size = [int(x) for x in grid_size[1:-1].split(',')]
        self.work_size = [self.grid_size[i] * self.block_size[i] for i in range(3)]

    def add_metric(self, section_name: str, metric_name: str, value: str, unit: str):
        if self.metrics is None:
            self.metrics = {}
        if self.metrics_str is None:
            self.metrics_str = {}
        if self.metrics_units is None:
            self.metrics_units = {}
        if self.metrics_sections is None:
            self.metrics_sections = {}
        self.metrics_str[metric_name] = value
        try:
            self.metrics[metric_name] = float(value.replace(',', ''))
        except ValueError:
            self.metrics[metric_name] = 0

        self.metrics_units[metric_name] = unit
        self.metrics_sections[metric_name] = section_name


class InterpretNvidiaCsv:
    def get_clean_name(self, name_mangled: str):
        match = re.search(r"llmsycl::kernels::(\w+)", name_mangled)
        name_clean = ""
        if match:
            name_clean = match.group(1)
            if name_clean.find("Kernel") != -1:
                name_clean = name_clean.replace("Kernel", "")
        else:
            if name_mangled.find("cutlass") != -1:
                name_clean = name_mangled[name_mangled.index("<"): name_mangled.index(">")]
            else:
                if name_mangled.find("_kernel") != -1:
                    name_clean = name_mangled.split("_")[0]
                else:
                    if name_mangled.find("_sgemm_") != -1:
                        name_clean = name_mangled
                    else:
                        print("Could not find clean name for kernel: ", name_mangled)
        return name_clean.lower()

    def exist_kernel_name(self, kernel_name: str) -> bool:
        return kernel_name in list(self.profiled_detailed.keys())

    def exist_kernel_id(self, kernel_name: str, kernel_id: int) -> bool:
        if not self.exist_kernel_name(kernel_name):
            return False
        return kernel_id in list(self.profiled_detailed[kernel_name].keys())

    def __init__(self, file_path: str):
        with open(file_path, 'r') as f:
            self.reader = csv.reader(f, delimiter=',', quotechar='"')
            self.rows = []
            next_is_header = False
            past_header = False
            self.profiled_detailed: dict[str: {int: DetailedEntry}] = {}  # kernel_name -> {launch_id: DetailedEntry}
            for row in self.reader:
                self.rows.append(row)
                if past_header:
                    name_clean = self.get_clean_name(row[self.resolve_index_by_header_name("Kernel Name")])
                    _id = int(row[self.resolve_index_by_header_name("ID")])

                    if not self.exist_kernel_name(name_clean):
                        self.profiled_detailed[name_clean] = {}

                    if not self.exist_kernel_id(name_clean, _id):
                        self.profiled_detailed[name_clean][_id] = \
                            DetailedEntry(
                                str(_id),
                                row[self.resolve_index_by_header_name("Stream")],
                                row[self.resolve_index_by_header_name("Context")],
                                row[self.resolve_index_by_header_name("Process Name")],
                                row[self.resolve_index_by_header_name("Device")],
                                row[self.resolve_index_by_header_name("CC")],
                                row[self.resolve_index_by_header_name("Block Size")],
                                row[self.resolve_index_by_header_name("Grid Size")]
                            )

                    if self.exist_kernel_id(name_clean, _id) and row[
                        self.resolve_index_by_header_name("Metric Name")] != "":
                        self.profiled_detailed[name_clean][_id].add_metric(
                            row[self.resolve_index_by_header_name("Section Name")],
                            row[self.resolve_index_by_header_name("Metric Name")],
                            row[self.resolve_index_by_header_name("Metric Value")],
                            row[self.resolve_index_by_header_name("Metric Unit")]
                        )

                if next_is_header:
                    if row[0].find("==ERROR== ") == -1:
                        self.header = row
                        next_is_header = False
                        past_header = True
                    else:
                        print("Warning: The host program had returned a non zero exit code during profiling.")
                else:
                    if len(row) > 0 and past_header == False:
                        if row[0].find("==PROF== Disconnected from process") != -1:
                            next_is_header = True

    def resolve_index_by_header_name(self, header_name: str):
        return self.header.index(header_name)

    def group_by_kernel_names(self):
        return self.profiled_detailed