#!/usr/bin/env python
"""\
rssmqrv -- Ranked Score Sorted Margins Quota-based Reweighted Voting --\n\n

Given a CSV file of weighted score ballots, seat M winners with
Droop-proportional representation.

Each seat is chosen using Ranked Score Sorted Margins (a Condorcet completion method),
then the ballots are reweighted proportionally to their total score for the seat 
winner.

If score sum does not exceed a quota, top scores are successively elevated to maxscore
until either quota is exceeded or non-zero scores are exhausted.
"""
from csvtoballots import csvtoballots
from asm import sorted_margins, myfmt
import argparse
import sys
import os
import numpy as np
from math import log10

def droopquota(n,m):
    return(n/(m+1))

def RSSM(Score,A,cnames,verbose=0):
    """"The basic Ranked Score Sorted Margins method, starting from
    non-normalized scores and the pairwise array"""
    ranking = Score.argsort()[::-1] # Seed the ranking using Score
    sorted_margins(ranking,Score,(A.T > A),cnames,verbose=verbose)
    if verbose > 0:
        print('[RSSM] Winner vs. Runner-up pairwise result: ',
              '{}:{} >= {}:{}'.format(cnames[ranking[0]],
                                      myfmt(A[ranking[0],ranking[1]]),
                                      cnames[ranking[1]],
                                      myfmt(A[ranking[1],ranking[0]])))
        return(ranking)

def rssmqrv(ballots, weights, cnames, numseats, verbose=0):
    """Run RSSM to elect <numseats> winners in a Droop proportional multiwnner election"""
    numballots, numcands = np.shape(ballots)
    ncands = numcands

    numvotes = weights.sum()
    numvotes_orig = float(numvotes)  # Force a copy

    quota = droopquota(numvotes,numseats)

    maxscore = ballots.max()

    cands = np.arange(numcands)

    winners = []

    maxscorep1 = maxscore + 1

    beta = np.arange(maxscorep1) / maxscore

    runner_up = -1

    factor_array = []

    for seat in range(numseats+1):

        if verbose>0:
            if (seat < numseats):
                print("- "*30,"\nStarting count for seat", seat+1)
            else:
                print("- "*30,"\nStarting count for runner-up")
            print("Number of votes:",myfmt(numvotes))

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
            for r in range(1,maxscorep1):
                rscores = np.where(permballot==r,w,0)
                S[r]  += rscores
                A     += np.multiply.outer(rscores,np.where(permballot<r,1,0))

        for r in range(1,maxscorep1):
            Score += r * S[r]

        if verbose > 2:
            print("\nFull Pairwise Array, step", seat+1)
            print("     [ " + " | ".join(cnames[cands]) + " ]")
            for c, row in zip(cnames[cands],A):
                n = len(row)
                print(" {} [ ".format(c),", ".join([myfmt(x) for x in row]),"]")

        # Determine the seat winner using sorted margins elimination:
        permranking = RSSM(Score,A,cnames[cands],verbose=verbose)
        permwinner = permranking[0]
        winner = cands[permwinner]

        # Seat the winner, then eliminate from candidates for next count
        if (seat == numseats):
            runner_up = winner
        else:
            # Save pairwise votes for the most recent non-runner-up seat winner
            X_vs_Y = np.zeros((numcands))
            Y_vs_X = np.zeros((numcands))
            X_vs_Y[cands] = A[permwinner]
            Y_vs_X[cands] = A[...,permwinner]

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
        # NOTE:  fractions are adjusted to avoid division by maxscore as much as possible.
        winsum = Score[permwinner]
        winsum_description = "\tWinner's score % before reweighting:  {}%".format(myfmt((winsum/
                                                                                         maxscore/
                                                                                         numvotes_orig)*100))
        print("Winner's normalized score: ", myfmt(winsum / maxscore))
        factor = 0.0
        v = 0
        winscores = ballots[...,winner]
        scorerange = np.arange(maxscorep1)
        if winsum > quota*maxscore:
            # Where possible, use score-based scaling
            remove = quota/winsum
            weights = np.multiply(1. - winscores*remove, weights)
            factor = 1. - remove*maxscore
            factors = 1.0 - scorerange*remove
        else:
            # Otherwise, successively raise fractional scores to full approval
            # until the quota is reached
            ss = np.multiply(np.arange(maxscorep1),S[...,permwinner])

            winsum = 0
            v = 1
            for r in range(maxscore,0,-1):
                winsum = maxscore*S[r:,permwinner].sum() + ss[:r].sum()
                if winsum > quota*maxscore:
                    v = r
                    break

            if winsum > quota*maxscore:
                remove = quota/winsum
                factor = 1. - remove*maxscore
                weights = np.multiply(1. - np.where(winscores>=v,maxscore,winscores)*remove, weights)
                factors = 1.0 - np.where(scorerange<v,scorerange,maxscore)*remove
            else:
                weights = np.where(ballots[...,winner]>0, 0, weights)
                factors = np.zeros((maxscorep1))

        numvotes = weights.sum()

        factor_array.append(list(factors))

        if verbose:
            print("Winner's votes per rating: ",
                  (", ".join(["{}:{}".format(j,myfmt(f))
                              for j, f in zip(scorerange[-1:0:-1],
                                              S[-1:0:-1,permwinner])])))
            print("After reweighting ballots:")
            print("\tQuota:  {}%".format(myfmt(quota/numvotes_orig*100)))
            print(winsum_description)
            if (v > 0):
                print("\t*** Winner {}'s score below quota.".format(cnames[winner]))
                print("\t*** Backup score:  {}%, after elevating rates >= {}".format(myfmt((winsum/
                                                                                           maxscore/
                                                                                           numvotes_orig)*100),v))
            print("\tReweighting factor per rating:  ", end="")
            if (factor > 0):
                print(", ".join(["{}:{}".format(j,myfmt(f))
                                 for j, f in zip(scorerange[-1:0:-1],
                                                 factors[-1:0:-1])]))
            else:
                print(factor)

            print("\tPercentage of vote remaining after reweighting:  {}%".format(myfmt((numvotes/
                                                                                        numvotes_orig) * 100)))
            if seat == numseats:
                print("\tSeat {} winner vs. Runner-up in Seat {} contest:".format(seat,seat))
                w_name = cnames[winners[-1]]
                r_name = cnames[winner]
                w_votes = X_vs_Y[winner]
                r_votes = Y_vs_X[winner]
                if w_votes > r_votes:
                    comp_sign = ">"
                elif w_votes < r_votes:
                    comp_sign = "<"
                else:
                    comp_sign = "=="
                print("\t{}:{} {} {}:{}".format(w_name,
                                                myfmt(w_votes),
                                                comp_sign,
                                                r_name,
                                                myfmt(r_votes)))

        if (numvotes <= (quota + numvotes_orig/1000.) ):
            break

    if verbose > 1 and numseats > 1:
        print("- "*30 + "\nReweighting factors for all seat winners:")
        for w, factors in zip(winners,factor_array):
            print(" {} |".format(cnames[w]),
                  ", ".join(["{}:{}".format(j,myfmt(f)) for j, f in zip(scorerange[1:],
                                                                        factors[1:])]))

    return(winners, runner_up)

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-i", "--inputfile",
                        type=str,
                        required=True,
                        help="REQUIRED: CSV Input file [default: none]")
    parser.add_argument("-m", "--seats", type=int,
                        default=1,
                        help="Number of seats [default: 1]")
    parser.add_argument("-t", "--filetype", type=int,
                        default=0,
                        help="CSV file type.  0 = Scores, 1 = RCV [default: 0]")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0,
                        help="Add verbosity [default: 0]")

    args = parser.parse_args()

    ballots, weights, cnames = csvtoballots(args.inputfile,ftype=args.filetype)

    print("- "*30)
    print("RANKED SCORE, SORTED MARGINS, QUOTA-BASED REWEIGHTED VOTING (RSSMQRV)")
    if args.verbose > 3:
        print("- "*30)
        # Figure out the width of the weight field, use it to create the format
        ff = '\t{{:{}d}}:'.format(int(log10(weights.max())) + 1)

        print("Ballots:\n\t{}, {}".format('weight',','.join(cnames)))
        for ballot, w in zip(ballots,weights):
            print(ff.format(w),ballot)

    winners, runner_up = rssmqrv(ballots, weights, cnames, args.seats, verbose=args.verbose)
    print("- "*30)
    
    if args.seats == 1:
        winfmt = "1 winner"
    else:
        winfmt = "{} winners".format(args.seats)

    print("\nRSSMQRV returns {}:".format(winfmt),", ".join([cnames[q] for q in winners]))

    if runner_up >= 0:
        print("Runner-up: ", cnames[runner_up])

if __name__ == "__main__":
    main()
