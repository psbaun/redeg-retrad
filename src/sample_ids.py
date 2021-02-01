#!/usr/bin/env python
"""
generate list of random post ids for 4chan data set 

data: path to list of post indices for entire data
nsize: sample size
seed: seed for random number generator
"""
import re
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data", required=True, help="Path to the ids list")
    ap.add_argument("-n", "--nsize", required=True, type=int, help="Sample size")
    ap.add_argument("-s", "--seed", required=False, default=1234, type=int)
    args = vars(ap.parse_args())

    with open(args["data"], "r") as fname:
        ids = fname.read().split()
    np.random.seed(args["seed"])
    idxs = np.random.choice(len(ids) + 1, args["nsize"], replace=False)
    outname = re.sub(".txt", "_n{}.txt".format(args["nsize"]), args["data"])
    np.savetxt(outname, idxs, fmt="%i", newline="\n")

if __name__=="__main__":
    main()