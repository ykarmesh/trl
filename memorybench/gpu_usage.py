#!/usr/bin/env python
import argparse
import re
import subprocess
from collections import defaultdict


def get_lab_caps():
    # Get caps per lab
    lab_caps = {}
    limits = (
        subprocess.run(["sacctmgr", "-n", "show", "qos", "-P"], stdout=subprocess.PIPE)
        .stdout.decode("utf-8")
        .splitlines()
    )
    for line in limits:
        if "lab" in line:
            fields = line.split("|")
            tres = fields[9].split(",")
            lab_caps[fields[0]] = defaultdict(int)
            for res in tres:
                caps = res.split("=")
                lab_caps[fields[0]][caps[0].replace("gres/gpu:", "")] = caps[1]
    return lab_caps


def print_lab_totals(lines):
    lab_caps = get_lab_caps()
    all_gpus = [
        "a40",
        "l40s",
        "rtx_6000",
        "a5000",
        "titan_x",
        "2080_ti",
        "cpu",
        "total_gpus",
    ]
    lab_caps["overcap"] = {k: 0 for k in lab_caps[list(lab_caps.keys())[0]]}
    lab_caps["scavenger"] = {k: 0 for k in lab_caps[list(lab_caps.keys())[0]]}
    # Add total_gpus to lab_caps
    for lab in lab_caps:
        lab_caps[lab]["total_gpus"] = sum(
            [int(lab_caps[lab][gpu]) for gpu in lab_caps[lab] if gpu != "cpu"]
        )
    lab_data = {}

    for line in lines:
        parts = line.split()
        if parts[2] == "RUNNING":
            # Find all occurrences of GPU types
            gpu_matches = re.findall(r"gres/gpu:(\w+)", line)
            lab = parts[-1]
            if lab not in lab_data:
                lab_data[lab] = {}
                lab_data[lab]["total_gpus"] = 0
                for gpu in gpu_matches:
                    lab_data[lab][gpu] = 0
                lab_data[lab]["cpu"] = 0

            # Add CPU data
            cpus = int(re.search(r"cpu=(\d+)", line).group(1))
            lab_data[lab]["cpu"] += cpus

            # Extract GPU information and update user data
            gpu_info = re.findall(r"gres/gpu=(\d+),gres/gpu:(\w+)=\d+", line)
            for count, gpu_type in gpu_info:
                count = int(count)
                if gpu_type not in lab_data[lab]:
                    lab_data[lab][gpu_type] = 0
                lab_data[lab][gpu_type] += count
                lab_data[lab]["total_gpus"] += count

    # Print results
    columns = ["lab".center(16)] + [str(gpu).center(11) for gpu in all_gpus]
    header = "  ".join(columns)
    print("Legend: <in_use> / <partition_cap>")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    total_used = defaultdict(int)
    total_cap = defaultdict(int)

    max_used_widths = {gpu: 0 for gpu in all_gpus}
    max_lab_cap_widths = {gpu: 0 for gpu in all_gpus}
    for lab, gpu_data in lab_data.items():
        for gpu in all_gpus:
            max_used_widths[gpu] = max(
                max_used_widths[gpu], len(str(gpu_data.get(gpu, 0)))
            )
            max_lab_cap_widths[gpu] = max(
                max_lab_cap_widths[gpu], len(str(lab_caps[lab][gpu]))
            )

    for lab, gpu_data in sorted(
        lab_data.items(), key=lambda x: x[1]["total_gpus"], reverse=True
    ):
        row = []
        # Check if the lab is using at least one GPU
        if gpu_data["total_gpus"] > 0:
            row = [lab.rjust(16)]
            for gpu in all_gpus:
                used_str = str(gpu_data.get(gpu, 0)).rjust(max_used_widths[gpu])
                cap_str = str(lab_caps[lab][gpu]).rjust(max_lab_cap_widths[gpu])
                if len(used_str) < 4 and len(cap_str) < 4:
                    row.append(f"{used_str} / {cap_str}".center(11))
                else:
                    row.append(f"{used_str}/{cap_str}".center(11))
                total_used[gpu] += gpu_data.get(gpu, 0)
                total_cap[gpu] += int(lab_caps[lab][gpu])

            print("  ".join(row))

    print("-" * len(header))

    row = ["total".center(16)]
    for gpu in all_gpus:
        used_str = str(total_used[gpu]).rjust(max_used_widths[gpu])
        cap_str = str(total_cap[gpu]).rjust(max_lab_cap_widths[gpu])
        if len(used_str) < 4 and len(cap_str) < 4:
            row.append(f"{used_str} / {cap_str}".center(11))
        else:
            row.append(f"{used_str}/{cap_str}".center(11))
    print("  ".join(row))


def print_user_totals(lines, filtered_partition):
    # Filter out non-RUNNING jobs and extract GPU types
    running_jobs = []
    gpu_types = {"cpu"}

    for line in lines:
        parts = line.split()
        if parts[2] == "RUNNING":
            running_jobs.append(parts)
            # Find all occurrences of GPU types
            gpu_matches = re.findall(r"gres/gpu:(\w+)", line)
            for match in gpu_matches:
                gpu_types.add(match)

    # Sort GPU types by VRAM capacity
    vram = {
        "a40": 46068,
        "rtx_6000": 24576,
        "a5000": 24564,
        "titan_x": 12288,
        "2080_ti": 11264,
    }
    sorted_gpu_types = sorted(
        list(gpu_types), key=lambda x: vram.get(x, 0), reverse=True
    )

    # Initialize user data structure
    user_data = {}
    for job in running_jobs:
        user, tres_alloc, partition = job[0], job[1], job[3]
        is_overcap = partition in ("overcap", "scavenger")

        # Initialize user entry if not present
        if user not in user_data:
            user_data[user] = {
                gpu: [0, 0, 0] for gpu in sorted_gpu_types
            }  # [non-overcap, overcap, total]
            user_data[user]["total_gpus"] = 0

        # Extract GPU information and update user data
        gpu_info = re.findall(r"gres/gpu=(\d+),gres/gpu:(\w+)=\d+", tres_alloc)
        for count, gpu_type in gpu_info:
            count = int(count)
            user_data[user][gpu_type][0 if not is_overcap else 1] += count
            user_data[user][gpu_type][2] += count
            user_data[user]["total_gpus"] += count

        cpus = int(re.search(r"cpu=(\d+)", tres_alloc).group(1))
        user_data[user]["cpu"][0 if not is_overcap else 1] += cpus
        user_data[user]["cpu"][2] += cpus

    user_width = max(len(user) for user in user_data)

    if filtered_partition == "all":
        legend = "Legend: <lab+overcap usage> (<lab-usage>, <overcap-usage>)"
    else:
        legend = "Legend: <lab usage>"

    # Create and populate the PrettyTable
    columns = (
        ["username".rjust(user_width)]
        + [
            str(gpu).center(16) if filtered_partition == "all" else f"{gpu: >8}"
            for gpu in sorted_gpu_types
        ]
        + ["total_gpus".rjust(11)]
    )
    header = " ".join(columns)
    print(legend)
    print("-" * max(len(legend), len(header)))
    print(header)
    print("-" * max(len(legend), len(header)))

    for user, user_data in sorted(
        user_data.items(), key=lambda x: x[1]["total_gpus"], reverse=True
    ):
        # Check if the user is using at least one GPU
        if user_data["total_gpus"] > 0:
            row = [user.rjust(user_width)]
            for gpu in sorted_gpu_types:
                non_overcap, overcap, total = user_data[gpu]

                if filtered_partition == "all":
                    row.append(f"{total: >4} " + f"({non_overcap}/{overcap})".ljust(11))
                else:
                    row.append(f"{user_data[gpu][2]}".rjust(8))

            row.append(f"{user_data['total_gpus']}".rjust(11))
            print(" ".join(row))


def main(args):
    # Running the specified command using subprocess
    command_output = subprocess.check_output(
        ["squeue", "-O", "UserName:20,tres-alloc:100,State,Partition"],
        text=True,
    )

    # Process the command output
    lines = command_output.strip().split("\n")[1:]  # Skip the header line
    lines = [
        line for line in lines if args.partition in line or args.partition == "all"
    ]  # Filter by partition

    if args.view == "user":
        print_user_totals(lines, args.partition)
    elif args.view == "lab":
        print_lab_totals(lines)
    else:  # both
        print("\n=== Lab GPU Usage ===")
        print_lab_totals(lines)
        print("\n=== User GPU Usage ===")
        print_user_totals(lines, args.partition)


if __name__ == "__main__":
    # parse args and get the partition
    parser = argparse.ArgumentParser(
        description="GPU usage reporting tool for lab and user statistics", add_help=True
    )
    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        help="Show only users under specific partition. Note, this will not be able to show overcap usage",
        required=False,
        default="all",
    )
    parser.add_argument(
        "-v",
        "--view",
        type=str,
        choices=["lab", "user", "both"],
        help="Choose which view to display: lab (per-lab statistics), user (per-user statistics), or both",
        required=False,
        default="both",
    )
    args = parser.parse_args()
    main(args)