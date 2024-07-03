import pandas as pd
import argparse
import csv
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def read_tidy_csv(fname):
    # parse the csv file with the csv package
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        data = list(reader)
        header = data[0]
        data = data[1:]
        return header, data


def get_clean_name(name_mangled: str):
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
                if name_mangled.find("sgemm_") != -1:
                    name_clean = name_mangled
                else:
                    print("Could not find clean name for kernel: ", name_mangled)
    print("%s -> %s" % (name_mangled, name_clean.lower()))
    return name_clean.lower()


def parse_vtune_csv(fname):
    header, data = read_tidy_csv(fname)
    index_name = header.index('Computing Task')
    # index_wsize = header.index('Work Size:Global')
    index_time = header.index('Computing Task:Total Time')

    d = {}
    for row in data:
        clean_name = get_clean_name(row[index_name])
        print(clean_name, row[index_time])
        if clean_name not in d:
            d[clean_name] = []
        # if row[index_wsize] not in d[clean_name]:
        #    d[clean_name][row[index_wsize]] = []
        d[clean_name].append(row[index_time])

    print(d.keys())
    total_time_per_kernel_non_la = 0
    total_time_per_kernel_la = 0
    for k in d:
        if k != '':
            sum = 0
            for i in d[k]:
                sum += float(i)
            if k.find('gemm') != -1:
                total_time_per_kernel_la += sum
            else:
                total_time_per_kernel_non_la += sum
    print(total_time_per_kernel_la)
    print(total_time_per_kernel_non_la)
    return total_time_per_kernel_la, total_time_per_kernel_non_la


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--csv', type=str, required=True)
    args = argparse.parse_args()
    print(parse_vtune_csv(args.csv))
