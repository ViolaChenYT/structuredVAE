#!/usr/bin/env python3
"""
Script to list all path names from the path dictionary.
This helps Snakemake discover which paths need to be processed.
"""

import json
import gzip
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python list_paths.py <paths_dict_file>")
        sys.exit(1)
    
    paths_dict_file = sys.argv[1]
    
    # Load path dictionary
    opener = gzip.open if paths_dict_file.endswith(".gz") else open
    with opener(paths_dict_file, "rt") as f:
        path_dict = json.load(f)
    
    # Print all path names
    for path_name in sorted(path_dict.keys()):
        print(path_name)

if __name__ == "__main__":
    main()
