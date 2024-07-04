import argparse
import pickle
import pathlib
from helpers.parse_ncu_csv import InterpretNvidiaCsv
from helpers.gather_metrics1 import ncu_gather_metrics
from helpers.process_vtune_csv_brief import parse_vtune_csv


def add_metadata_to_data(
        data,
        is_ncu_cuda: bool,
        is_ncu_sycl: bool,
        is_vtune_sycl: bool):
    obj = {
        'is_ncu_cuda': is_ncu_cuda,
        'is_ncu_sycl': is_ncu_sycl,
        'is_vtune_sycl': is_vtune_sycl,
        'gpu': input("Enter the GPU model: "),
        'data': data
    }
    return obj


if __name__ == '__main__':
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--ncu', type=str, required=False)
    argparse.add_argument('--vtune', type=str, required=False)
    argparse.add_argument('--out', type=str, required=True)
    argparse.add_argument('--type', type=str, required=True)
    args = argparse.parse_args()

    if args.ncu is None and args.vtune is None:
        print("Need an NCU or a VTune csv to work with.")
        exit(1)
    if args.ncu is not None and args.vtune is not None:
        print("Can only handle one type of csv at a time.")
        exit(2)

    valid_types = ['ncu_cuda', 'ncu_sycl', 'vtune_sycl']
    if args.type not in valid_types:
        print("Invalid type. Must be one of: ", valid_types)
        exit(3)

    if args.ncu is not None:
        raw = InterpretNvidiaCsv(args.ncu)
        grouped = raw.group_by_kernel_names()
        with open(args.out, 'wb') as f:
            pickle.dump(
                add_metadata_to_data(grouped, args.type == 'ncu_cuda', args.type == 'ncu_sycl', False),
                f
            )
        print("Parsed NCU csv and saved to ", args.out)

    if args.vtune is not None:
        la_nonla_total = parse_vtune_csv(args.ncu)
        with open(args.out, 'wb') as f:
            pickle.dump(
                add_metadata_to_data(la_nonla_total, False, False, args.type == 'vtune_sycl'),
                f
            )
        print("Parsed VTune csv and saved to ", args.out)
