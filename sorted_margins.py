#!/usr/bin/env python
from collections import deque
import numpy as np
from math import log10

def myfmtscalar(x):
    if x > 1:
        fmt = "{:." + "{}".format(int(log10(x))+5) + "g}"
    else:
        fmt = "{:.5g}"
    return fmt.format(x)

def myfmt(x):
    try:
        c = myfmtscalar(x)
    except:
        c = "(" + ", ".join([myfmtscalar(y) for y in x]) + ")"
    return c

def mydiff(f,t):
    "Return the difference between f (from) and t (to), as scalar or tuple"
    try:
        md = tuple([d - c for c, d in zip(f,t)])
    except:
        md = t - f
    return md

def smith_from_losses(losses,cands):
    sum_losses = losses.sum(axis=1)
    min_losses = sum_losses.min()

    # Initialize Smith set and queue
    smith = set(np.compress(sum_losses==min_losses,cands))
    queue = deque(smith)

    # Loop until queue is empty
    while (len(queue) > 0):
        # pop first item on queue
        c = queue.popleft()
        beats_c = np.compress(losses[c],cands)
        # for each candidate who defeats current candidate in Smith,
        # add that candidate to Smith and stick them on the end of the
        # queue
        for d in beats_c:
            if d not in smith:
                smith.add(d)
                queue.append(d)
    return(smith)

def sorted_margins(ranking,metric,loss_array,cnames,verbose=0):
    """
    Performs sorted margins using a generic metric seed, based on a
    loss_array like A.T > A
    ranking is a pre-sorted, numpy array
    """
    nswaps = 0
    # Loop until no pairs are out of order pairwise
    n = len(ranking)
    ncands = len(metric)
    mmin = metric[0]
    mmax = metric[0]
    for m in metric[1:]:
        if m < mmin:
            mmin = m
        if m > mmax:
            mmin = m
    maxdiff = mydiff(mmin,mmax)
    if verbose > 1:
        print(". "*30)
        print("Showing Sorted Margin iterations, starting from seeded ranking:")
        print('\t{}\n'.format(' > '.join(['{}:{}'.format(cnames[c],myfmt(metric[c])) for c in ranking])))
    while True:
        apprsort = [metric[r] for r in ranking]
        apprdiff = []
        outoforder = []
        mindiffval = maxdiff
        mindiff = ncands
        for i in range(1,n):
            im1 = i - 1
            c_i = ranking[i]
            c_im1 = ranking[im1]
            if (loss_array[c_im1,c_i]):
                outoforder.append((c_im1,c_i))
                apprdiff.append(mydiff(apprsort[i],apprsort[im1]))
                if apprdiff[-1] < mindiffval:
                    mindiff = im1
                    mindiffval = apprdiff[-1]
        # terminate when no more pairs are out of order pairwise:
        if (len(outoforder) == 0) or (mindiff == ncands):
            break

        # Do the swap
        nswaps += 1
        ranking[range(mindiff,mindiff+2)] = ranking[range(mindiff+1,mindiff-1,-1)]

        if verbose > 1:
            print("Out-of-order candidate pairs, with margins:")
            for k, pair in enumerate(outoforder):
                c_im1, c_i = pair
                name_im1 = cnames[c_im1]
                name_i = cnames[c_i]
                print('\t{} < {}, margin = {}'.format(name_im1,name_i,myfmt(apprdiff[k])))
            print('\nSwap #{}: swapping candidates {} and {} with minimum margin {}'.format(
                  nswaps,cnames[ranking[mindiff]],cnames[ranking[mindiff+1]],myfmt(mindiffval)))
            print('\t{}\n'.format(' > '.join([cnames[c] for c in ranking])))

    if verbose > 1:
        print("\tCandidates in pairwise order")

    if verbose > 0:
        smith = smith_from_losses(np.where(loss_array, 1, 0),np.arange(ncands))
        print(". "*30)
        if len(smith) == 1:
            print("[SORTED MARGINS] Pairwise winner == Sorted Margins winner: ", cnames[ranking[0]])
        else:
            print("[SORTED MARGINS] No pairwise winner; Smith set:", [cnames[c] for c in ranking if c in smith], "-- Sorted Margins winner:", cnames[ranking[0]])

    return
