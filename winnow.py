#!/usr/bin/env python
"""\
Run a primary election to winnow candidates. \n

Uses Score ballots. Non-zero score indicates approval.  Voters may optionally indicate
a top preference cutoff. This is relevant for Preference Approval Sorted Margins.

Repeat until a threshold of votes are used up:
\t- Accumulate pairwise vote array, scores, and approval+preference.
\t- Determine winner (see below)
\t- After each winner is found, exhaust ballots giving non-zero
\t  score to the winner, either (default) exhausting winner-approving ballots entirely, or
\t  (optionally, using -x/--max50 flag) exhausting a max of 50% of remaining ballots.
\t- Continue until (default) 7 candidates have been advanced, or there are no more
\t  candidates with approval above the exclusive approval threshold.

Default winner selection method:
\t- Preference-Approval Sorted Margins with explicit preference above Preference cutoff.\n
Alternatives: 
\t- Preference, Approval, Top Rating, Score, Vote 3-2-1, STAR, Score Sorted Margins.

Default exclusive approval threshold 1/100 (1%)\n

NB:\tVote 3-2-1 is generalized to accommodate full score ballots with explicit Preference
\tcutoff. Non-zero score is considered Approved. Above Preference Cutoff (either per ballot
\tor above '-c CUTOFF' score level) is considered Preferred. Candidates are sorted in
\tdescending order of Preference, and the top three are selected. Next, the least approved
\tof those three is eliminated. Finally, the most preferred pairwise of the top two is the
\tV321 winner. While default preference cutoff for other methods is zero, for V321 the default
\tcutoff is MAXSCORE - 1.  If the cutoff is set to zero, V321 is just Top-Two Approval instant
\trunoff.
"""
import argparse
from ballot_tools.csvtoballots import *
import numpy as np
import sorted_margins as sm
import tabulate
from ssmpr import ssm, SmartDescriptionFormatter
from asm import find_dcindex

myfmt = sm.myfmt

def winnow(ballots,
           weights,
           cnames,
           cands,
           numseats,
           method=0,
           invthreshlevel=32,
           cutoff=None,
           dcindex=-1,
           max50=False,
           verbose=0):
    """Singe Winner winnowing election with score ballots"""

    numballots, numcands = np.shape(ballots)
    ncands = int(numcands) # force copy

    if (numseats > ncands):
        print("*** WARNING, # seats > # candidates! ***")
        print("Resetting # seats to", ncands)
        numseats = int(numcands)

    numvotes = weights.sum()
    numvotes_orig = float(numvotes)  # Force a copy

    threshlevel = numvotes / invthreshlevel

    maxscore = ballots.max()

    if dcindex < 0:
        if cutoff == None:
            if method == 5:             # V321
                cutoff = maxscore - 1
            else:
                cutoff = 0

    winners = []
    overall_approval = []
    exclusive_approval = []
    exhausted_votes = []

    maxscorep1 = maxscore + 1

    scorerange = np.arange(maxscorep1)

    numseatsm1 = numseats - 1

    Total_Approval = np.zeros((ncands))

    for seat in range(numseats):

        if verbose>0:
            print("- "*30,"\nStarting count for advancing candidate #{}".format(seat+1))
            print("Number of votes remaining:",myfmt(numvotes))

        # ----------------------------------------------------------------------
        # Tabulation:
        # ----------------------------------------------------------------------
        # During Tabulation, restrict qualified candidates to those with approval
        # above the threshold level.
        (ncands,
         cands,
         S,
         A,
         Pref,
         Approval,
         Score ) = tabulate.pairwise_pref_approval_scores(ballots,
                                                          weights,
                                                          cands,
                                                          cnames,
                                                          maxscore,
                                                          maxscorep1,
                                                          threshlevel=threshlevel,
                                                          cutoff=cutoff,
                                                          dcindex=dcindex,
                                                          verbose=verbose)
        if (seat == 0):
            Total_Approval[cands] = Approval        # unpermute total approval

        if ncands == 0:

            if verbose:
                print("Halting: No more qualified candidates above {}% approval".format(myfmt(100/invthreshlevel)))
            break

        elif ncands == 1:
            # Handle only-one-quota-threshold-qualifying-candidate case:
            permwinner = 0
            winner = cands[0]
            permranking = np.array([cands[0]])
            if verbose > 1:
                print("Only one candidate left:", cnames[winner])
        else:                               # ncands > 1
            TopRating = S[maxscore,...]

            permrating = {0:list(zip(Pref,Approval,TopRating,Score)), # PASM
                          1:list(zip(Pref,Approval,TopRating,Score)), # Preference
                          2:list(zip(Approval,TopRating,Pref,Score)), # Approval
                          3:list(zip(TopRating,Pref,Approval,Score)), # TopRating
                          4:list(zip(Score,TopRating,Approval,Pref)), # Score
                          5:list(zip(Pref,Approval,TopRating,Score)), # V321
                          6:list(zip(Score,TopRating,Approval,Pref)), # STAR
                          7:list(zip(Score,TopRating,Approval,Pref))  # SSM
                          }[method]
            permlabel  = {0:"(Pref,Approval,TopRating,Score)", # PASM
                          1:"(Pref,Approval,TopRating,Score)", # Preference
                          2:"(Approval,TopRating,Pref,Score)", # Approval
                          3:"(TopRating,Pref,Approval,Score)", # TopRating
                          4:"(Score,TopRating,Approval,Pref)", # Score
                          5:"(Pref,Approval,TopRating,Score)", # V321
                          6:"(Score,TopRating,Approval,Pref)", # STAR
                          7:"(Score,TopRating,Approval,Pref)"  # SSM
                          }[method]

            permranking = np.array(sorted(np.arange(ncands),
                                          key=(lambda c:permrating[c]),
                                          reverse=True))

            if verbose:
                print("Candidate,", permlabel)
                for c in permranking:
                    print(cnames[cands[c]], permrating[c])

            if (method < 5):                          # PASM, Preference, Approval, TopRating, Score
                if (method == 0):                           # PASM
                    sm.sorted_margins(permranking,
                                      permrating,
                                      (A.T > A),
                                      cnames[cands],
                                      verbose=verbose)
                permwinner = permranking[0]
            elif (method == 5):                         # V321
                # Drop the lowest approved of the top 3 candidates already sorted by Preference
                # but keep them in preference-sorted order.
                v20,v21 = sorted(sorted(permranking[:3],
                                        key=(lambda c:Approval[c]),
                                        reverse=True)[:2],
                                 key=(lambda c:permrating[c]),
                                 reverse=True)
                # Of the remaining top two, find the pairwise winner:
                permwinner = int(v20)
                if A[v21,v20] > A[v20,v21]:
                    # The original ordering is only upset if the less-preferred
                    # of the two makes a clear defeat of the other.
                    # If it's a tie, the most-preferred wins.
                    permwinner = int(v21)

                if verbose > 1:
                    print("Vote321:\n\tTop 3 candidates by preference:")
                    print("\t{}".format(", ".join([cnames[cands[c]] for c in permranking[:3]])))
                    print("\n\tThe two highest approved of those three are:")
                    print("\t{}, {}".format(cnames[cands[v20]], cnames[cands[v21]]))
                    print("\n\tPairwise winner between those two:", cnames[cands[permwinner]])

            elif method == 6 or method == 7:            # STAR / SSM
                permwinner = ssm(permranking,permrating,A,cnames[cands],verbose=verbose)
                if method == 7:                         # SSM
                    permwinner = permranking[0]
                # else, permwinner = STAR_winner

        try:
            winner = cands[permwinner]
        except:
            print("permwinner: ", permwinner, ", permranking: ", [cands[c] for c in permranking])

        if verbose:
            print("\n-----------\n*** Seat {}: {}\n-----------\n".format(seat+1,cnames[winner]))

        winners += [winner]
        overall_approval += [Total_Approval[winner]]
        exclusive_approval += [Approval[permwinner]]

        cands = np.compress(cands != winner,cands)
        ncands -= 1
        winscores = ballots[...,winner]
        numvotes_old = float(numvotes)

        # Figure out how much to take out of ballots that approve of the winner.
        # A maximum of 50% of remaining votes is removed for each advanced winner.
        winner_approval = Approval[permwinner]
        if verbose:
            print("Winner's exclusive approval: {} votes, {}%".format(myfmt(winner_approval),
                                                                      myfmt(winner_approval/numvotes_orig*100)))
        if (not max50) or ((2*winner_approval) < numvotes):
            factor = 0.0
            if verbose:
                print("Reweighting: all ballots approving {} are exhausted".format(cnames[winner]))
        else:
            factor = 1. - numvotes / winner_approval / 2.
            if verbose:
                print("Reweighting: all ballots approving {} are rescaled by".format(cnames[winner]),
                      myfmt(factor*100), "percent")

        weights = np.where(winscores>0, weights*factor, weights)
        numvotes = weights.sum()
        percent_exclusive = (numvotes_old - numvotes) / numvotes_orig * 100.
        percent_represented = (numvotes_orig - numvotes) / numvotes_orig * 100.
        percent_remaining = numvotes / numvotes_orig * 100.
        exhausted_votes += [numvotes_old - numvotes]

        if verbose:
            print("Winner's votes per rating: ",
                  (", ".join(["{}:{}".format(j,myfmt(f))
                              for j, f in zip(scorerange[-1:0:-1],
                                              S[-1:0:-1,permwinner])])))
            print("After reweighting ballots:")
            print("\t% of exclusive vote for this winner: {}%".format(myfmt(percent_exclusive)))
            print("\t% of vote represented at this step:  {}%".format(myfmt(percent_represented)))
            print("\t% of vote remaining:                 {}%".format(myfmt(percent_remaining)))

        if ncands == 0:
            if verbose > 0:
                print("Halting:  No more candidates remaining")
            break

    return(winners,overall_approval,exclusive_approval,exhausted_votes)

def main():
    from math import log10
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=SmartDescriptionFormatter)

    methods = ["PASM", "P", "A", "TR", "S", "V321", "STAR", "SSM"]

    method_description = ["PREFERENCE APPROVAL SORTED MARGINS",
                          "PREFERENCE",
                          "APPROVAL",
                          "TOP RATING",
                          "SCORE",
                          "VOTE 3-2-1",
                          "SCORE THEN AUTOMATIC RUNOFF",
                          "SCORE SORTED MARGINS"]

    parser.add_argument("-i", "--inputfile",
                        type=str,
                        required=True,
                        help="REQUIRED: CSV Input file [default: none]")
    parser.add_argument("-c", "--cutoff",
                        type=int,
                        required=False,
                        default=None,
                        help="Approval cutoff rating [default: None]")
    parser.add_argument("-d", "--deprecated_candidate",
                        type=str,
                        required=False,
                        default=None,
                        help=("Preference cutoff candidate name. "
                        "Any candidate at or below the ballot's score for "
                        "deprecated_candidate is not preferred. [default: None]"))
    parser.add_argument("-l", "--inverse_threshold_level",
                        type=int,
                        default=100,
                        help=("Drop candidates below the inverse of this level. "
                        "For example, if level is 100, drop candidates below 1/100 "
                        "(1%%) approval. [default: 100]"))
    parser.add_argument("-m", "--numseats",
                        type=int,
                        default=7,
                        help="Maximum number of candidates to advance past winnowing [default: 7]")
    parser.add_argument("-s", "--select_method",
                        type=str,
                        choices=methods,
                        default=methods[0],
                        help=("Select method: " + "; ".join(["{} ({})".format(m,d)
                                                             for m, d in zip(methods,method_description)])
                                                + " [default: PASM]"))
    parser.add_argument("-t", "--filetype", type=str,
                        choices=["score", "rcv"],
                        default="score",
                        help="CSV file type, either 'score' or 'rcv' [default: 'score']")
    parser.add_argument("-x", "--max50",
                        action='store_true',
                        default=False,
                        help="""Exhausting a maximum of 50 percent of remaining ballots for each candidate
                        advanced [default: False]""")
    parser.add_argument("--rerun",
                        action='store_true',
                        default=False,
                        help="Rerun method with just the first round winners [default: False]")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0,
                        help="Add verbosity (increase by repetition) [default: 0]")

    args = parser.parse_args()

    ftype={'score':0, 'rcv':1}[args.filetype]

    ballots, weights, cnames = csvtoballots(args.inputfile,ftype=ftype)
    numballots, numcands = np.shape(ballots)
    cands = np.arange(numcands)

    dcindex = find_dcindex(cnames,dcname=args.deprecated_candidate)

    print("- "*30)

    methargs = np.arange(len(methods))

    try:
        method = dict(zip(methods,methargs))[args.select_method]
    except:
        print("Error, invalid method selected")
        return

    print(method_description[method])

    if args.verbose > 3:
        print("- "*30)
        # Figure out the width of the weight field, use it to create the format
        ff = '\t{{:{}d}}:'.format(int(log10(weights.max())) + 1)

        print("Ballots:\n\t{}, {}".format('weight',','.join(cnames)))
        for ballot, w in zip(ballots,weights):
            print(ff.format(w),ballot)

    (winners,
     overall_approval,
     exclusive_approval,
     exhausted_votes) = winnow(ballots,
                               weights,
                               cnames,
                               cands,
                               args.numseats,
                               method=method,
                               invthreshlevel=args.inverse_threshold_level,
                               cutoff=args.cutoff,
                               dcindex=dcindex,
                               max50=args.max50,
                               verbose=args.verbose)

    print("- "*30)

    numseats = len(winners)
    if numseats == 1:
        winfmt = "1 winner"
    else:
        winfmt = "{} winners".format(numseats)

    print("\n{} results, ".format(methods[method]), end="")
    print("{}:".format(winfmt),", ".join([cnames[q] for q in winners]))

    if args.verbose:
        print("\n\n{:30}{:20}{:20}{:20}\n".format("Winner,",
                                                  "Overall Approval,",
                                                  "Exclusive Approval,",
                                                  "Exhausted Votes"))
        for w, oa, xa, xv in zip(winners,overall_approval,exclusive_approval,exhausted_votes):
            print("{:30}{:20}{:20}{:20}".format(cnames[w]+",",
                                                str(myfmt(oa))+",",
                                                str(myfmt(xa))+",",
                                                str(myfmt(xv))))

    if args.rerun:
        print("\n", "- "*30, "\nRe-running for single winner with only winnowed candidates:\n")
        (rw, oa, xa, xv) = winnow(ballots,
                                  weights,
                                  cnames,
                                  np.array(winners),
                                  1,
                                  method=method,
                                  invthreshlevel=args.inverse_threshold_level,
                                  cutoff=args.cutoff,
                                  dcindex=dcindex,
                                  max50=args.max50,
                                  verbose=args.verbose)
        print("Final winner: ", cnames[rw[0]])

if __name__ == "__main__":
    main()
