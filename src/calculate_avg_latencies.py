import argparse
import csv
from os.path import join, pardir, exists, isdir
from os import listdir


def average_latencies(results_dir, files_prefix, filename):
    assert exists(results_dir), f"Results directory ('{results_dir}') doesn't exist."
    assert isdir(
        results_dir
    ), f"Results directory ('{results_dir}') is not a directory."
    filepath = join(results_dir, filename)
    assert not exists(filepath), f"Output file ('{filepath}') already exists."

    files = [
        join(results_dir, f) for f in listdir(results_dir) if f.startswith(files_prefix)
    ]

    total_inferences = 0
    model_time = {}
    all_models = set()
    all_filters = set()
    for f in files:
        with open(f) as file_in:
            reader = csv.DictReader(file_in)
            first_line = True
            for line in reader:
                if first_line:
                    total_inferences += int(line["total_inferences"])
                    first_line = False
                filters = line["filters"]
                all_filters.add(filters)
                for key in line:
                    if key.endswith(".pickle"):
                        all_models.add(key)
                        model_time[filters, key] = model_time.get(
                            (filters, key), 0
                        ) + float(line[key])

    all_models = sorted(all_models)
    all_filters = sorted(all_filters)

    with open(filepath, "w") as file_out:
        file_out.write(f"{total_inferences=},{','.join(all_models)}\n")
        for f in all_filters:
            file_out.write(f)
            for m in all_models:
                file_out.write(f",{model_time[f, m] / total_inferences:.20f}")
            file_out.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script averages out the inference time from multiple runs of the measure_latency script."
    )

    parser.add_argument(
        "-r",
        "--results_dir",
        help="path to the results directory created by the training_models script",
        default=join(pardir, "out"),
    )
    parser.add_argument(
        "-p",
        "--prefix",
        help="common prefix of the filenames that contain latencies to be averaged",
        default="inference_latency",
    )
    parser.add_argument(
        "-o",
        "--filename",
        help="filename to export the average latency results",
        default="average_inference_latency.csv",
    )

    args = parser.parse_args()

    subdirs = [f for f in listdir(args.results_dir) if isdir(join(args.results_dir, f))]
    print("Found following subdirs:", ", ".join(subdirs))
    for sd in subdirs:
        print(f"Averaging latencies inside '{sd}' subdir...")
        average_latencies(
            join(args.results_dir, sd, "models"), args.prefix, args.filename
        )
