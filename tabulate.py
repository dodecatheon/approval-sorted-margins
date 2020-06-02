#!/usr/bin/env python
from sorted_margins import myfmt
import numpy as np

def scores(ballots,weights,cands,maxscore,maxscorep1,quota=0.0,verbose=0):
    """
    Tabulate ballots to find scores, then compress to only active candidates.
    Also determine where score exceeds the quota

    Returns:
      PermS:  the total approval sums at each rating level given candidates in cands
      PermT:  total approval sums at and above each rating level for candidates in cands
      PermScores: Total weighted scoresums within the top quota of ballots for each candidiate
      qtar: quota threshold approval rating for positions in cands

    
    """
    numballots, numcands = np.shape(ballots)
    if quota == 0.0:
        quota = weights.sum()
    ncands = len(cands)
    # S[r,x]: Total votes for candidate x at rating r
    S = np.zeros((maxscorep1,numcands))
    T = np.zeros((maxscorep1,numcands))
    # ----------------------------------------------------------------------
    # Tabulation:
    # ----------------------------------------------------------------------
    for ballot, w in zip(ballots,weights):
        for r in range(1,maxscorep1):
            rscores = np.where(ballot==r,w,0)
            S[r]  += rscores
    
    PermS = S[...,cands]
    PermQ = np.zeros((maxscorep1,ncands))
    Total = np.zeros((ncands))
    STotal = np.zeros((ncands))
    inds = np.arange(ncands)
    qtar = np.zeros((ncands),dtype=int)
    PermScore = np.zeros((ncands))
    for r in range(maxscore,0,-1):
        Srow = PermS[r]
        Total += Srow
        # qtar: Quota Threshold Approval Rating
        # For those candidates whose Total exceeds the quota for the first time,
        # set qtar for that candidate to the current rating.
        qtar = np.where(qtar==0,np.where(Total>quota,r,qtar),qtar)

        for i in inds:
            if qtar[i] < r:
                PermQ[r,i] = 1.0
                PermScore[i] += r*Srow[i]
                STotal[i] += r*Srow[i]
            elif qtar[i] == r:
                score_r_i = Srow[i] - (Total[i] - quota)
                PermQ[r,i] = score_r_i / Srow[i]
                PermScore[i] += r*score_r_i
                STotal[i] += r*Srow[i]

    return(PermS,PermQ,PermScore,STotal,qtar)

def pairwise(ballots,weights,cands,Q,cnames,maxscore,maxscorep1,verbose=0):
    numballots, numcands = np.shape(ballots)
    ncands = len(cands)
    # A: pairwise array, equal-rated-none
    A = np.zeros((numcands,numcands))
    # S[r,x]: Total votes for candidate x at rating r
    S = np.zeros((maxscorep1,numcands))
    # ----------------------------------------------------------------------
    # Tabulation:
    # ----------------------------------------------------------------------
    rscores = np.zeros((numcands))
    for ballot, w in zip(ballots,weights):
        for r in range(1,maxscorep1):
            rscores = np.where(ballot==r,w,0.)
            rscores[cands] *= Q[r]
            A     += np.multiply.outer(rscores,np.where(ballot<r,1,0))

    PermA = np.zeros((ncands,ncands))
    for i,c in enumerate(cands):
        PermA[i] = A[c][cands]

    if (verbose > 2):
        print("\nFull Pairwise Array")
        print("     [ " + " | ".join(cnames[cands]) + " ]")
        for c, row in zip(cnames[cands],PermA):
            print(" {} [ ".format(c),", ".join([myfmt(x) for x in row]),"]")

    return(PermA)
