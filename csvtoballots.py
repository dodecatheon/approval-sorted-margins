#!/usr/bin/env python
# % matplotlib notebook
# import matplotlib
# % config InlineBackend.figure_format = 'svg'
# % matplotlib inline
# import matplotlib.pyplot as plt
import numpy as np
from math import *
import csv
import re
# from pprint import pprint

def csvtoballots(filename):
    "convert csv file to a set of ballots"
    # Read the csv file's cnames to find number of candidates
    with open(filename,"r") as csvfile:
        # Automatically detect the delimiter of the file:
        try:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
        except Exception as e:
            print(filename,": ERROR can't Sniffer().sniff file, exception =", e, file=sys.stderr)
            sys.exit(1)

        csvfile.seek(0)

        reader = csv.reader(csvfile,dialect)

        delimiter = dialect.delimiter

        # First row is cnames
        cnames = next(reader)
        for i, name in enumerate(cnames):
            cnames[i] = name.strip()

        ncand = len(cnames)

    if cnames[0].startswith('weight'):
        weight = np.genfromtxt(filename,
                               dtype=int,
                               skip_header=1,
                               usecols=(0),
                               delimiter=delimiter,
                               filling_values=0)

        ballots = np.genfromtxt(filename,
                                dtype=int,
                                skip_header=1,
                                usecols=range(1,len(cnames)),
                                delimiter=delimiter,
                                filling_values=0)

        cnames = cnames[1:]
    else:
        ballots = np.genfromtxt(filename,
                                dtype=int,
                                skip_header=1,
                                delimiter=delimiter,
                                filling_values=0)

        weight = np.ones((len(ballots),),dtype=int)

    return(ballots, weight, cnames)

if __name__ == "__main__":
    import sys

    if (len(sys.argv) == 2):
        fname = sys.argv[1]
    else:
        fname = input("Enter csv filename: ")

    ballots, weights, cnames = csvtoballots(fname)

    if len(cnames) == 0:
        cnames = [str(i) for i in range(len(ballots[0]))]

    print(" {}:".format("weight"),cnames)
    for w, ballot in zip(weights,ballots):
        print("{}:".format(w),ballot)
