#!/usr/bin/env python
"""\
Run a primary election to winnow candidates. \n
\n
Uses Score ballots. Non-zero score indicates approval.
Voters may optionally indicate a top preference cutoff. This is relevant for
Preference Approval Sorted Margins.\n
\n
Repeat until a threshold of votes are used up:\n
\n
\t- Accumulate pairwise vote array, scores, and approval+preference.\n
\n
\n- Default method:\n
\t- Preference-Approval Sorted Margins with explicit preference above Preference cutoff.\n
\t- Alternatives:\n 
\t- Preference//Approval, Score, STAR, Score Sorted Margins.\n
\t- After each winner is found, apply exhaustive reweighting to ballots giving non-zero\n
\t  score to the winner, either (default) exhausting up to 50% of remaining total ballot weight, or\n
\t  reducing those ballots to zero weight to simulate a general election.\n
\t- Continue until (default) 7 candidates have been advanced, or there are no more\n
\t  candidates with approval above the threshold.
\n
Default exclusive approval threshold 1/100 (1%)\n
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
           numseats,
           method=0,
           invthreshlevel=32,
           cutoff=None,
           dcindex=-1,
           general=False,
           verbose=0):
    """Singe Winner winnowing election with score ballots"""

    if dcindex < 0:
        if cutoff == None:
            cutoff = 0

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

    cands = np.arange(numcands)

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
            print("- "*30,"\nStarting count for winnowing candidates", seat+1)
            print("Number of votes:",myfmt(numvotes))

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
            if (method < 2):                # PASM & Approval
                permrating = list(zip(Pref,Approval,S[maxscore,...],Score))
                permlabel = "(Preference, Approval, TopRating, Score)"
            elif (method < 5):              # Score, STAR, SSM
                permrating = list(zip(Score,S[maxscore,...],Approval,Pref))
                permlabel = "(Score, Toprating, Approval, Preference)"

            permranking = np.array(sorted(np.arange(ncands),
                                          key=(lambda c:permrating[c]),
                                          reverse=True))

            if (method == 0):                           # PASM
                sm.sorted_margins(permranking,
                                  permrating,
                                  (A.T > A),
                                  cnames[cands],
                                  verbose=verbose)
                permwinner = permranking[0]
            elif (method < 3):                          # Pref/Approval, Score
                permwinner = permranking[0]
            elif method == 3 or method == 4:            # STAR / SSM
                permwinner = ssm(permranking,permrating,A,cnames[cands],verbose=verbose)
                if method == 4:                         # SSM
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
        if general or ((2*winner_approval) < numvotes):
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

    methods = ["PASM", "PA", "Score", "STAR", "SSM"]

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
    parser.add_argument("-g", "--general",
                        action='store_true',
                        default=False,
                        help="Run like a general election, exhausting all approved ballots [default: False]")
    parser.add_argument("-l", "--inverse_threshold_level",
                        type=int,
                        default=100,
                        help=("Drop candidates below the inverse of this level. "
                        "For example, if level is 100, drop candidates below 1/100 "
                        "(1%%) approval. [default: 100]"))
    parser.add_argument("-m", "--numseats",
                        type=int,
                        default=7,
                        help="Number of candidates advancing to general elecction [default: 7]")
    parser.add_argument("-s", "--select_method",
                        type=str,
                        choices=methods,
                        default=methods[0],
                        help=("Select method: PASM (Preference Approval Sorted Margins, "
                        "a Condorcet method); PA (Preference//Approval); Score; STAR "
                        "(Score Then Automatic Runoff); or SSM (Score Sorted Margins). "
                        "[default: PASM]"))
    parser.add_argument("-t", "--filetype", type=str,
                        choices=["score", "rcv"],
                        default="score",
                        help="CSV file type, either 'score' or 'rcv' [default: 'score']")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0,
                        help="Add verbosity [default: 0]")

    args = parser.parse_args()

    ftype={'score':0, 'rcv':1}[args.filetype]

    ballots, weights, cnames = csvtoballots(args.inputfile,ftype=ftype)

    dcindex = find_dcindex(cnames,dcname=args.deprecated_candidate)

    print("- "*30)

    methargs = np.arange(len(methods))

    try:
        method = dict(zip(methods,methargs))[args.select_method]
    except:
        print("Error, invalid method selected")
        return

    method_description = ["PREFERENCE APPROVAL SORTED MARGINS",
                          "PREFERENCE//APPROVAL",
                          "SCORE",
                          "STAR = Score Then Automatic Runoff, with SMV weighting",
                          "SCORE SORTED MARGINS"]

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
                               args.numseats,
                               method=method,
                               invthreshlevel=args.inverse_threshold_level,
                               cutoff=args.cutoff,
                               dcindex=dcindex,
                               general=args.general,
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
        print("Winner, Overall Approval, Exclusive Approval, Exhausted Votes")
        for w, oa, xa, xv in zip(winners,overall_approval,exclusive_approval,exhausted_votes):
            print("\t{},\t{},\t{},\t{}".format(cnames[w],myfmt(oa),myfmt(xa),myfmt(xv)))

if __name__ == "__main__":
    main()
