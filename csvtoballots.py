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

def csvtoballots(filename, ftype=0):
    "convert csv file to a set of ballots"
    with open(filename,"r") as csvfile:
        # Automatically detect the delimiter of the file:
        try:
            dialect = csv.Sniffer().sniff(csvfile.read(1024))
        except Exception as e:
            print(filename,": ERROR can't Sniffer().sniff file, exception =", e, file=sys.stderr)
            sys.exit(1)

        csvfile.seek(0)

        delimiter = dialect.delimiter
        reader = csv.reader(csvfile, quotechar='"', delimiter=delimiter,
                            quoting=csv.QUOTE_ALL, skipinitialspace=True)

        if ftype == 0:
	        # Default filetype is scores:
	        # First line is [weight,]<Comma-separated list of candidate names>
	        # Subsequent lines are [weight,]<Comma-separated scores>
  
	        # Read the csv file's cnames to find number of candidates

	        # First row is cnames
            cnames = np.array(next(reader))
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

                weight = np.ones((len(ballots)),dtype=int)

        elif ftype == 1:
            # Filetype 1 is Ranked Choice Voting:
            # First line is <comma separated list of choices>
            # Subsequent lines are comma-separated lists of candidate names with
            # "undervote" or "overvote" for a spoiled or unfilled ranking
            candset = set()
            top = next(reader)
            numballots = 0
            for row in reader:
                numballots += 1
                nr = len(row)
                for r in row:
                    candset.add('{}'.format(r))

            nocountset = set(['overvote','undervote'])
            candset -= nocountset

            cnames = np.array(sorted([c for c in candset]))

            candindex = dict((c,i) for i, c in enumerate(cnames))
            ncands = len(cnames)

            ballots = np.zeros((numballots,ncands),dtype='int')
            csvfile.seek(0)
            top = next(reader)
            for row,b in zip(reader,ballots):
	            for j, r in enumerate(row):
                    rr = '{}'.format(r)  # standardize quote formatting
                    if rr not in nocountset:
                        b[candindex[rr]] = 10 - j
            weight = np.ones((len(ballots)),dtype=int)

    return(ballots, weight, cnames)

if __name__ == "__main__":
    import sys

    if (len(sys.argv) == 2):
        fname = sys.argv[1]
    else:
        fname = input("Enter csv filename: ")

    ballots, weights, cnames = csvtoballots(fname)

    if len(cnames) == 0:
        cnames = np.array([str(i) for i in range(len(ballots[0]))])

    print(" {}:".format("weight"),cnames)
    for w, ballot in zip(weights,ballots):
        print("{}:".format(w),ballot)
