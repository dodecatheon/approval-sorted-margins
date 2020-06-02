#!/usr/bin/env python3
"""
Approval Sorted Margins.  

Runs an election on rated ballots.

By default, the explicit approval cutoff is (maxscore - 1)/2 rounded down to nearest integer. 

Condorcet-full//Condorcet-approved//Approval and Smith//Approval winners are included for comparison.
"""
import numpy as np
from ballot_tools.csvtoballots import *
from sorted_margins import sorted_margins, myfmt, smith_from_losses

def asm(ballots, weight, cnames, cutoff=None, verbose=0):
    "Approval Sorted Margins"

    nballots, ncands = np.shape(ballots)
    cands = np.arange(ncands)
    tw = weight.sum()

    maxscore = int(ballots.max())
    maxscorep1 = maxscore + 1

    if cutoff == None:
        cutoff = (maxscore - 1) // 2

    minapprove = cutoff + 1
    # ----------------------------------------------------------------------
    # Tabulation:
    # ----------------------------------------------------------------------
    # A: pairwise array, equal-rated-none
    # B: Same as A, but only including ratings above cutoff
    # T:  Total approval for candidate X
    A = np.zeros((ncands,ncands))
    B = np.zeros((ncands,ncands))
    T = np.zeros((ncands))
    for ballot, w in zip(ballots,weight):
        for r in range(1,maxscorep1):
            A += np.multiply.outer(np.where(ballot==r,w,0),
                                   np.where(ballot<r ,1,0))
        for r in range(minapprove,maxscorep1):
            B += np.multiply.outer(np.where(ballot==r,w,-1),
                                   np.where(ballot<r ,1,0))
            T += np.where(ballot==r,w,0)

    # ----------------------------------------------------------------------
    # Rank and rating calculations:
    # ----------------------------------------------------------------------
    # Find the Smith set (the set of candidates that defeats all candidates
    # outside the set)
    smith = smith_from_losses(np.where(A.T > A, 1, 0),cands)
    nsmith = len(smith)
    bsmith = smith_from_losses(np.where(B.T > B, 1, 0),cands)
    nbsmith = len(bsmith)

    # Determine Approval Sorted Margins winner:
    # Seed the list by descending approval:
    ranking = np.array([c for c in T.argsort()[::-1] if c in smith])
    branking = np.array([c for c in T.argsort()[::-1] if c in bsmith])

    sorted_margins(ranking,T,(A.T > A),cnames,verbose=verbose)
    winner = ranking[0]

    # Put approval on diagonals
    A += np.diag(T)
    B += np.diag(T)

    return(winner,tw,ncands,maxscore,cutoff,ranking,branking,T,A,B)
 
def test_asm(ballots,weight,cnames,cutoff=None,verbose=0):
    
    winner,tw,ncands,maxscore,cutoff,ranking,branking,T,A,B = asm(ballots,weight,cnames,cutoff=cutoff,verbose=0)

    cands = np.arange(ncands)
    nsmith = len(ranking)
    nbsmith = len(branking)
    approval_ranking = T.argsort()[::-1]

    print("\nFull Pairwise Array, approval on diagonal, cutoff @ {}:".format(cutoff))
    for row in A:
        print(row)

    if (nsmith > 1) and (nbsmith == 1):
        print("\nApproval-only Pairwise Array, approval on diagonal, cutoff @ {}:".format(cutoff))
        for row in B:
            print(row)

    Margins = np.multiply.outer(T,np.ones((ncands))) - np.multiply.outer(np.ones((ncands)),T)
    Margins = np.where((A.T>A),0,Margins)

    print("\nApproval rankings:")
    print("\t{}".format(' > '.join([cnames[c] for c in approval_ranking])))

    print("\nSmith set, ranked by Approval Sorted Margins:")
    print("\t{}".format(' > '.join([cnames[c] for c in ranking])))

    print("\nSmith set for approved ratings only, ranked by approval:")
    print("\t{}".format(' > '.join([cnames[c] for c in branking])))

    if nsmith > 1:
        print("\nApproval Sorted Margins pairwise results (on Smith Set):")
        for i in range(1,nsmith):
            im1 = i - 1
            c_i = ranking[i]
            c_im1 = ranking[im1]
            cname_i = cnames[c_i]
            cname_im1 = cnames[c_im1]
            print("\t{}>{}: {} > {}".format(cname_im1,cname_i,A[c_im1,c_i],A[c_i,c_im1]))

    print("\nASM Winner: ", cnames[winner])

    # Condorcet-full//Condorcet-approved//Approval winner
    if nsmith == 1:
        bwinner = ranking[0]
    elif nbsmith == 1:
        bwinner = branking[0]
    else:
        bwinner = approval_ranking[0]

    print("\nC-full//C-approved//Approval Winner: ", cnames[bwinner])

    # Smith//Approval Winner
    if nsmith == 1:
        sawinner = ranking[0]
    else:
        sawinner = [c for c in approval_ranking if c in ranking][0]

    print("\nSmith//Approval Winner: ", cnames[sawinner])

    print("-----\n")
    # Also print out non-Smith pairwise
    full_ranking = T.argsort()[::-1]
    sorted_margins(full_ranking,T,(A.T > A),cnames,verbose=verbose)
    winner = full_ranking[0]
    print("All candidates, ranked by Approval Sorted Margins:")
    print("\t{}".format(' > '.join([cnames[c] for c in full_ranking])))
    print("\nApproval Sorted Margins pairwise results for all candidates:")
    for i in range(1,ncands):
        im1 = i - 1
        c_i = full_ranking[i]
        c_im1 = full_ranking[im1]
        cname_i = cnames[c_i]
        cname_im1 = cnames[c_im1]
        print("\t{}>{}: {} > {}".format(cname_im1,cname_i,A[c_im1,c_i],A[c_i,c_im1]))
    print("-----")

    return

def main():
    import argparse
    from math import log10
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("-i", "--inputfile",
                        type=str,
                        required=True,
                        help="REQUIRED: CSV Input file")
    parser.add_argument("-c", "--cutoff",
                        type=int,
                        required=False,
                        default=None,
                        help="Approval cutoff [default: None]")
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

    test_asm(ballots, weight, cnames, cutoff=args.cutoff, verbose=args.verbose)

if __name__ == "__main__":
    main()
