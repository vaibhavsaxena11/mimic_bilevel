"""
Helper script to merge multiple collect play data into one hdf5 file.

Example usage:
    change the 'read_folder' to location of hdf5's.

    python scripts/merge_datasets.py --dataset_dir=/Users/vaibhav/work/robomimicV2/LIBERO/datasets/libero_10
"""

import h5py
import argparse
import pathlib

def copy_attributes(source, target):
    """Copy attributes from source to target"""
    for key, value in source.attrs.items():
        target.attrs[key] = value

def main(args):

    # Path to your folder containing the hdf5 files
    read_folder = pathlib.Path(args.dataset_dir)
    # Path to new hdf5
    save_file = read_folder.parent / f"{read_folder.name}.hdf5"

    # List all hdf5 files in the directory
    hdf5_files = read_folder.glob("*.hdf5")

    counter = 0

    # Create or open the merged.hdf5 file
    with h5py.File(str(save_file), "w") as merged_file:
        # Create a group named data if it doesn't exist yet
        data_group = merged_file.require_group("data")

        # Iterate over all the hdf5 files and merge demos
        for hdf5_file in hdf5_files:
            with h5py.File(str(hdf5_file), 'r') as source_file:
                source_data_group = source_file['data']
                if counter == 0:
                    copy_attributes(source_data_group, data_group)

                # Iterate through all demos in the 'data' group of the source file
                for demo_name in source_data_group:
                    new_demo_name = f"demo_{counter}"

                    # Copy demo to merged_file
                    source_file.copy(f"data/{demo_name}", data_group, new_demo_name)

                    # Copy attributes of the demo dataset
                    copy_attributes(source_data_group[demo_name], data_group[new_demo_name])

                    counter += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="folder containing hdf5 datasets",
    )

    args = parser.parse_args()

    main(args)

    print("Merging completed!")