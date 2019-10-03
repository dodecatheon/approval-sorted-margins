#!/usr/bin/env python
"""\
qssmrv -- Quota-based Ranked Score Sorted Margins Reweighted Voting --\n\n

Given a CSV file of weighted score ballots, seat M winners with
Droop-proportional representation.

Each seat is chosen using Ranked Score Sorted Margins (a Condorcet completion method),
then the ballots are reweighted proportionally to their score for the seat 
winner.
"""
from csvtoballots import csvtoballots
from asm import sorted_margins, myfmt
import argparse
import sys
import os
import numpy as np

def droopquota(n,m):
    return(n/(m+1))

def rssmqrv(ballots, weights, cnames, numseats, verbose=0):

    numballots, numcands = np.shape(ballots)
    ncands = numcands

    numvotes = sum(weights)
    weights_orig = np.array(weights)
    numvotes_orig = sum(weights_orig)

    quota = droopquota(numvotes,numseats)

    maxscore = ballots.max()

    cands = np.arange(numcands)

    winners = []

    maxscorep1 = maxscore + 1

    beta = np.arange(maxscorep1) / maxscore

    for seat in range(numseats+1):

        # ----------------------------------------------------------------------
        # Tabulation:
        # ----------------------------------------------------------------------
        # A: pairwise array, equal-rated-none
        # S[r,x]: Total votes for candidate x at rating r
        A = np.zeros((ncands,ncands))
        S = np.zeros((maxscorep1,ncands))
        Score = np.zeros((ncands))
        T = np.zeros((ncands))
        for ballot, w in zip(ballots,weights):
            permballot = ballot[cands]
            for r in range(maxscore,0,-1):
                rscores = np.where(permballot==r,w,0)
                S[r]  += rscores
                A     += np.multiply.outer(rscores,np.where(permballot<r,1,0))

        for r in range(maxscore,-1,-1):
            Score += r * S[r]

        # Determine the seat winner using sorted margins elimination:
        # Seat the winner, then eliminate from candidates for next count

        ranking = Score.argsort()[::-1] # Seed the ranking using Score
        sorted_margins(ranking,Score,(A.T > A),cnames[cands],verbose=verbose)
        permwinner = ranking[0]
        winner = cands[permwinner]

        if verbose:
            if (seat < numseats):
                print("\n-----------\n*** Seat {}: {}\n-----------\n".format(seat+1,cnames[winner]))
            else:
                print("\n-----------\n*** Runner-up: {}\n-----------\n".format(cnames[winner]))

        if (seat < numseats):
            winners += [winner]
            cands = np.compress(cands != winner,cands)
            ncands -= 1

        # Scale weights by proportion of Winner's score that needs to be removed
        winsum = Score[permwinner] / maxscore
        winsum_description = "\tWinner's score % before reweighting: {}%".format(myfmt((winsum/numvotes_orig)*100))
        factor = 0.0
        v = 0
        if winsum >= quota:
            # Where possible, use score-based scaling
            remove = quota/winsum

            for r in range(1,maxscorep1):
                factor = 1.0 - beta[r]*remove
                weights = np.where(ballots[:,winner]==r, factor*weights, weights)

        else:
            # Otherwise, total scaled score is less than one quota, so default to Bucklin scaling:
            winsum = 0
            v = 1
            for r in range(maxscore,0,-1):
                winsum += S[r][permwinner]
                if winsum > quota:
                    v = r
                    break

            if winsum >= quota:
                factor = (1.0 - (quota/winsum))
                weights = np.where(ballots[:,winner]>=v, factor*weights, weights)
            else:
                weights = np.where(ballots[:,winner]>0, 0, weights)

        numvotes = weights.sum()

        if verbose:
            print("After reweighting ballots:")
            print("\tQuota: {}%".format(myfmt(quota/numvotes_orig*100)))
            print(winsum_description)
            if (v > 0):
                print("\t*** Winner {}'s score below quota. ".format(cnames[winner]))
                print("\t*** Backup score: {}% rate winner >= {}".format(myfmt(winsum/numvotes_orig*100),v))
            print("\tReweighting factor:", myfmt(factor))
            print("\tPercentage of vote remaining after reweighting: {}%\n".format(myfmt((numvotes / numvotes_orig) * 100.)))

        if (numvotes <= (quota + numvotes_orig*0.001) ):
            break

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
    parser.add_argument('-v', '--verbose', action='count',
                        default=0,
                        help="Add verbosity [default: 0]")

    args = parser.parse_args()

    ballots, weights, cnames = csvtoballots(args.inputfile)

    winners = rssmqrv(ballots, weights, cnames, args.seats, verbose=args.verbose)

    print("=======================")
    print("RSSMQRV Winners = ",[cnames[q] for q in winners])

if __name__ == "__main__":
    main()
