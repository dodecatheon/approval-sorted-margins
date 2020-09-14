#!/usr/bin/env python
from sorted_margins import myfmt
import numpy as np

def scores(ballots,weights,cands,maxscore,quota=0.0,verbose=0):
    """
    Tabulate ballots to find scores, then compress to only active candidates.
    Also determine where score exceeds the quota

    Returns:
      PermS:  the total approval sums at each rating level given candidates in cands
      PermQ:  Fraction to be counted for votes at rating r by candidate i against candidates j=0..ncand
      PermScores: Total weighted scoresums within the top quota of ballots for each candidiate
      qtar: quota threshold approval rating for positions in cands

    
    """
    numballots, numcands = np.shape(ballots)
    if quota == 0.0:
        quota = weights.sum()
    ncands = len(cands)
    mp1 = maxscore + 1
    # S[r,x]: Total votes for candidate x at rating r
    S = np.zeros((mp1,numcands))
    # ----------------------------------------------------------------------
    # Tabulation:
    # ----------------------------------------------------------------------
    for ballot, w in zip(ballots,weights):
        for r in range(maxscore,0,-1):
            rscores = np.where(ballot==r,w,0)
            S[r]  += rscores
    
    PermS = S[...,cands]
    PermQ = np.zeros((mp1,ncands))
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

def pairwise(ballots,weights,cands,Q,cnames,maxscore,verbose=0):
    numballots, numcands = np.shape(ballots)
    ncands = len(cands)
    # A: pairwise array, equal-rated-none
    A = np.zeros((numcands,numcands))
    # S[r,x]: Total votes for candidate x at rating r
    # ----------------------------------------------------------------------
    # Tabulation:
    # ----------------------------------------------------------------------
    rscores = np.zeros((numcands))
    for ballot, w in zip(ballots,weights):
        for r in range(maxscore,0,-1):
            rscores = np.where(ballot==r,w,0.)
            # votes at rating r against other candidates are weighted by row r in Q
            # array.
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

def pairwise_scores(ballots,weights,cands,cnames,maxscore,maxscorep1,verbose=0):
    "Permuted Pairwise array and scores when quota is 100%"
    numballots, numcands = np.shape(ballots)
    ncands = len(cands)
    # A: pairwise array, equal-rated-none; A[x,y] = votes for cand x against cand y
    AA = np.zeros((numcands,numcands))
    # S[r,x]: Total votes at rating r for candidate x
    SS = np.zeros((maxscorep1,numcands))
    # ----------------------------------------------------------------------
    # Tabulation:
    # ----------------------------------------------------------------------
    rscores = np.zeros((numcands))
    for ballot, w in zip(ballots,weights):
        for r in range(maxscore,0,-1):
            rscores = np.where(ballot==r,w,0)
            SS[r]  += rscores
            AA     += np.multiply.outer(rscores,np.where(ballot<r,1,0))

    # Restrict from full ballot down to the specified candidates:
    S = SS[...,cands]

    # row permutation in NumPy is not quite as simple as column permutation:
    A = np.zeros((ncands,ncands))
    for i,c in enumerate(cands):
        A[i] = AA[c][cands]

    # Total score TS[x] = r*S[max,x] + ... + 2 * S[2,x] + 1 * S[1,x]
    # TS = np.zeros((ncands))
    # for r in range(maxscore,0,-1):
    #     TS += r * S[r]
    #
    # Using NumPy matrix-matrix multiplication:
    TS = np.array(np.matrix(np.diagflat(np.arange(maxscorep1))) * np.matrix(S)).sum(axis=0)
    TA = S.sum(axis=0)

    if (verbose > 2):
        print("\nFull Pairwise Array")
        print("     [ " + " | ".join(cnames[cands]) + " ]")
        for c, row in zip(cnames[cands],A):
            print(" {} [ ".format(c),", ".join([myfmt(x) for x in row]),"]")

    return(S,A,TS,TA)


def pairwise_pref_approval_scores(ballots,weights,cands,cnames,maxscore,maxscorep1,
                                  threshlevel=0.0,
                                  cutoff=0,
                                  dcindex=-1,
                                  verbose=0):
    "Permuted Pairwise array and scores when quota is 100%"
    numballots, numcands = np.shape(ballots)
    ncands = len(cands)
    rscores = np.zeros((numcands))
    # A: pairwise array, equal-rated-none; A[x,y] = votes for cand x against cand y
    AA = np.zeros((numcands,numcands))
    # S[r,x]: Total votes at rating r for candidate x
    SS = np.zeros((maxscorep1,numcands))
    # T: pairwise array for ties.
    TT = np.zeros((numcands,numcands))

    # Pref[x]: Total preference for candidate x
    PP = np.zeros((numcands))

    # ----------------------------------------------------------------------
    # Tabulation:
    # ----------------------------------------------------------------------

    if dcindex < 0:
        # Global preference cutoff, same for all ballots
        for ballot, w in zip(ballots,weights):
            for r in range(maxscore,0,-1):
                rscores = np.where(ballot==r,w,0)
                SS[r]  += rscores
                AA     += np.multiply.outer(rscores,np.where(ballot<r,1,0))
                TT     += np.multiply.outer(rscores,np.where(ballot==r,1,0))
            for r in range(maxscore,cutoff,-1):
                PP     += np.where(ballot==r,w,0)
    else:
        # Explicit preference cutoff, specified for each ballot
        for ballot, w in zip(ballots,weights):
            if ballot[dcindex] > 0:
                AA[dcindex,dcindex] += w
            for r in range(maxscore,0,-1):
                rscores = np.where(ballot==r,w,0)
                SS[r]  += rscores
                AA     += np.multiply.outer(rscores,np.where(ballot<r,1,0))
                TT     += np.multiply.outer(rscores,np.where(ballot==r,1,0))
        # Preference totals = votes against the Preference cutoff candidate
        PP = np.array(AA[...,dcindex])
        AA[dcindex,dcindex] = 0.0
    
    # Restrict from full ballot down to the specified candidates:
    S = np.zeros((0,0))
    Approval = np.zeros((0))
    A = np.zeros((0,0))
    Tied = np.zeros((0,0))
    Score = np.zeros((0))
    Pref = np.zeros((0))

    ncands_old = int(ncands)
    inds = np.arange(ncands)
    permqc = np.compress(SS[...,cands].sum(axis=0) > threshlevel,inds)
    ncands = len(permqc)

    if verbose and (ncands < ncands_old):
        print("Approval threshold reduces # candidates from {} to {}".format(ncands_old,ncands))

    if ncands == 0:
        return(ncands,cands,S,A,Pref,Approval,Score)
    cands = cands[permqc]
    S = SS[...,cands]
    Approval = S.sum(axis=0)

    # row permutation in NumPy is not quite as simple as column permutation:
    A = np.zeros((ncands,ncands))
    Tied = np.zeros((ncands,ncands))
    for i,c in enumerate(cands):
        A[i] = AA[c][cands]
        Tied[i] = TT[c][cands]

    # Total preference, restricted to just candidates:
    Pref = PP[cands]

    # Total score TS[x] = r*S[max,x] + ... + 2 * S[2,x] + 1 * S[1,x]
    # TS = np.zeros((ncands))
    # for r in range(maxscore,0,-1):
    #     TS += r * S[r]
    #
    # Using NumPy matrix-matrix multiplication:
    Score = np.array(np.matrix(np.diagflat(np.arange(maxscorep1))) * np.matrix(S)).sum(axis=0)

    if (verbose > 2):
        print("\nFull Pairwise Array")
        print("     [ " + " | ".join(cnames[cands]) + " ]")
        for c, row in zip(cnames[cands],A):
            print(" {} [ ".format(c),", ".join([myfmt(x) for x in row]),"]")

    return(ncands,cands,S,A,Pref,Approval,Tied,Score)

