#!/usr/bin/env python
"""\
PAIR Sorted Margins

\tP = Preferred
\tA = Approved / Acceptable
\tI = Insufficient / Inadequate
\tR = Reject

PAIR-SM runs a hybrid of Approval Sorted Margins and Score Sorted Margins on rated ballots.

There are 3 levels of "approval" above None / Reject: 

\t* Preferred gets a score of 1
\t* Approved/Acceptable/Adequate gets a score of 0.5
\t* Insufficient/Inadequate gets a score of 0.
\t* Reject / Blank gets a score of 0.

Ballots are tabulated to get pairwise counts and total approval ("PAIR") scores,
and an initial ordering of candidates is seeded by sorting in descending order
of each candidate's respective PAIR-score.

SORTED MARGINS PROCESS:
While any pairs of adjacent candidates are out of order pairwise, find the out-of-order
pair with the minimum PAIR-score margin, and swap them in the ordering.

The winner is the first candidate in the resulting ordering.

Minimum score range is 0-3. Ranges 0-3 to 0-9 are divided as follows:

    \tPreferred >>> Approved  >> Insuff.   > Reject
    \t--------- >>> --------- >> --------- > ---------
0-3:\t3         >>> 2         >> 1         > 0
0-4:\t4 > 3     >>> 2         >> 1         > 0
0-5:\t5 > 4     >>> 3 > 2     >> 1         > 0
0-6:\t6 > 5     >>> 4 > 3     >> 2 > 1     > 0
0-7:\t7 > 6 > 5 >>> 4 > 3     >> 2 > 1     > 0
0-8:\t8 > 7 > 6 >>> 5 > 4 > 3 >> 2 > 1     > 0
0-9:\t9 > 8 > 7 >>> 6 > 5 > 4 >> 3 > 2 > 1 > 0

In general, the top score of the Approved level is floor(Maxscore * 2 / 3),
while the top score of the Insufficient level is floor(Maxscore / 3)
"""
import numpy as np
from ballot_tools.csvtoballots import *
from sorted_margins import sorted_margins

def pairsm(ballots, weight, cnames, verbose=0):
    "Preferred/Acceptable/Insufficient/Reject Sorted Margins (PAIR-SM)"

    nballots, ncands = np.shape(ballots)
    cands = np.arange(ncands)
    tw = weight.sum()

    maxscore = int(ballots.max())
    nscores = maxscore + 1

    preferred_cutoff = (maxscore * 2)//3
    acceptable_cutoff = maxscore // 3

    pc = preferred_cutoff       # shortcut alias
    pcp1 = pc + 1

    ac = acceptable_cutoff      # shortcut alias
    acp1 = ac + 1

    # ----------------------------------------------------------------------
    # Tabulation setup:
    # ----------------------------------------------------------------------
    # A: pairwise array, equal-rated-whole
    # S: for each score, the total at that score for each candidate
    # TT : Preferred totals at 1 points each, Acceptable at 0.5 point, below that at zero.
    A  = np.zeros((ncands,ncands))
    S  = np.zeros((nscores,ncands))
    TT =  np.zeros((ncands))
    TP = np.zeros((ncands))
    TA = np.zeros((ncands))
    TI = np.zeros((ncands))
    Ties  = np.zeros((ncands,ncands))

    # ----------------------------------------------------------------------
    # Scores:
    # ----------------------------------------------------------------------
    for ballot, w in zip(ballots,weight):
        for r in range(maxscore,0,-1):
            A += np.multiply.outer(np.where(ballot==r,w,0),
                                   np.where(ballot<r,1,0))
            Ties += np.multiply.outer(np.where(ballot==r,w,0),
                                      np.where(ballot==r,1,0))
            S[r] += np.where(ballot==r,w,0)

    np.fill_diagonal(A,0)
    np.fill_diagonal(Ties,0)
    T = S.sum(axis=0)
    S[0] = tw - T
    TP = S[pcp1:,:].sum(axis=0)
    TA = S[acp1:pc,:].sum(axis=0)
    TI = S[1:ac,:].sum(axis=0)
    TT = (2*TP + TA) / 2.0

    # Determine Approval Sorted Margins winner:
    # Seed the list by descending PAIR points, with total Pref, total Pref+Approv, total Pref+Approv+Insuff as tie-breakers:
    rating = list(zip(TT,TP,TP+TA,TP+TA+TI))

    ranking = np.array(sorted(cands,key=(lambda c:rating[c]),reverse=True))
    if verbose:
        print("\nCandidate: (PAIRscore,TotPref,TotPref+TotAppr,TotPref+TotAppr+TotInsuff)")
        for c in ranking:
            print(cnames[c],":",rating[c])

    sorted_margins(ranking,TT,(A.T > A),cnames,verbose=verbose)
    winner = ranking[0]

    # Put PAIR score on diagonal
    A += np.diag(TT)

    return(winner,tw,ncands,maxscore,ranking,TT,Ties,A)
 
def test_pairsm(ballots,weight,cnames,verbose=0):
    
    (winner,
     tw,
     ncands,
     maxscore,
     ranking,
     TT,
     Ties,
     A) = pairsm(ballots,weight,cnames,verbose=verbose)

    cands = np.arange(ncands)

    print("\nFull Pairwise Array, PAIR score on diagonal:")
    for row in A:
        print(row)

    if np.any(Ties > 0):
        print("\nPairwise Ties Array:")
        for row in Ties:
            print(row)

    print("\nPAIR-SM ordering:")
    print("\t{}".format(' > '.join([cnames[c] for c in ranking])))

    print("\nPAIR-SM pairwise results:")
    for i in range(1,ncands):
        im1 = i - 1
        c_i = ranking[i]
        c_im1 = ranking[im1]
        cname_i = cnames[c_i]
        cname_im1 = cnames[c_im1]
        print("\t{}>{}: {} > {}".format(cname_im1,cname_i,A[c_im1,c_i],A[c_i,c_im1]))

    print("\nPAIR-SM Winner: ", cnames[winner])

    print("-----\n")

    return

def main():
    import argparse
    from ssmpr import SmartDescriptionFormatter
    from math import log10
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=SmartDescriptionFormatter)

    parser.add_argument("-i", "--inputfile",
                        type=str,
                        required=True,
                        help="REQUIRED: CSV Input file")
    parser.add_argument('-v',
                        '--verbose',
                        action='count',
                        default=0,
                        help="Add verbosity [default: 0]")
    args = parser.parse_args()

    ballots, weight, cnames = csvtoballots(args.inputfile)

    # Figure out the width of the weight field, use it to create the format
    ff = '\t{{:{}d}}:'.format(int(log10(weight.max())) + 1)

    print("Ballots:\n\t{}, {}".format('weight',','.join(cnames)))
    for ballot, w in zip(ballots,weight):
        print(ff.format(w),ballot)

    test_pairsm(ballots, weight, cnames,
                verbose=args.verbose)

if __name__ == "__main__":
    main()
