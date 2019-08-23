#!/usr/bin/env python
"""\
qrarv -- Quota-based Ranked Approval Reweighted Voting --\n\n

Given a CSV file of weighted score ballots, seat M winners with
proportional representation, using Sorted Margins Elimination to
determine the winner of each seat.
"""
from csvtoballots import csvtoballots
from smeminlv import sme_minlv
import argparse
import sys
import os
import numpy as np

def droopquota(n,m):
    return(n/(m+1))

def harequota(n,m):
    return(n/m)

def qrarv(ballots, weights, cnames, numseats, quotafunction):

    numballots, numcands = np.shape(ballots)

    numvotes = weights.sum()
    numvotes_orig = weights.sum()

    quota = quotafunction(numvotes,numseats)

    maxscore = ballots.max()

    cands = np.arange(numcands)

    verbose = (len(cnames)>0)

    winners = []

    maxscorep1 = maxscore + 1

    for seat in range(numseats):

        scoretotals = np.zeros((maxscorep1,numcands))

        # Calculate the pairwise array based on current weights
        for ballot, w in zip(ballots,weights):
            for v in range(maxscorep1):
                scoretotals[v] += np.where(ballot==v,w,0)

        # Determine the seat winner using sorted margins elimination:
        # Seat the winner, then eliminate from candidates for next count
        winner = sme_minlv(ballots,weights,cands,scalar=True)
        winners += [winner]
        cands = np.compress(cands != winner,cands)

        if verbose:
            print("-----------\n\tSeat {}: Winner = {}".format(seat,cnames[winner]))

        if (seat < (numcands - 1)):
            # find the score at which the winning candidate for this seat has
            # approval of at least a quota
            winsum = 0
            for k in reversed(range(maxscorep1)):
                winsum += scoretotals[k][winner]
                if winsum > quota:
                    v = k
                    break

            if winsum >= quota:
                factor = (1.0 - (quota/winsum))
            else:
                factor = 0.0

            weights = np.multiply(np.where(ballots[:,winner] >= v, factor, 1), weights)

            numvotes = weights.sum()

            if verbose:
                print("Scaling factor:", factor)

        if verbose:
            print("Remaining vote fraction: {:6.2f}%".format(numvotes / numvotes_orig * 100.))

    return(winners)

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-i", "--inputfile",
                        type=str,
                        required=True,
                        help="REQUIRED: CSV Input file [default: none]")
    parser.add_argument("-m", "--seats", type=int,
                        default=1,
                        help="Number of seats [default: 1]")
    parser.add_argument("-q", "--quota", type=str,
                        choices=["droop", "hare"],
                        default="droop",
                        help="Quota rule, where 'droop' quota is Nballots/(Nseats [default: droop]")

    args = parser.parse_args()

    quota_function = {"droop":droopquota, "hare":harequota}[args.quota]

    ballots, weights, cnames = csvtoballots(args.inputfile)

    winners = qrarv(ballots, weights, cnames, args.seats, quota_function)

    print("=======================")
    print("QRARV Winners = ",[cnames[q] for q in winners])

if __name__ == "__main__":
    main()
