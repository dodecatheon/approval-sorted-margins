#!/usr/bin/env python
"""
Score Sorted Margins quota-based reweighting PR voting --

Given a CSV file of weighted score ballots, seat M winners with
Hare quota.  If M is not specified, the Score sorted margins single
winner is found.

For "Droop" (actually Hagenbach-Bischhoff) quota, use M-1 of the winners with
the M-th winner as runner up.

Each seat is chosen using Score Sorted Margins (a Condorcet completion method),
then the ballots are reweighted proportionally to their scores for the seat
winner, based on the total approval at and above the quota threshold rating.

Note that the score used for each seat is taken within the top quota of votes,
as in Sequential Monroe. Here is how a top quota score is found for a particular candidate:

\t* Initialize the top quota score TQS[c] for candidate c to zero.
\t* Initialize the total approval TA[c] for candidate c to zero.

Starting at rating r = MaxScore, add weighted ballots scoring c at r to TA[c]:

\tTA[c] += S[r,c]

If TA[c] is less than the quota, add the appropriate amount of that score to TQS:

\tTQS[c] += r * S[r,c]

However, if the total approval at r now exceeds the quota, the top quota score receives
only that portion of the score corresponding to the approval above the quota:

\tTQS[c] += r * (S[r,c] - (TA[c] - quota))

This top quota score is the metric used for sorted margins. On the last seat (Hare quota),
the top quota score is the total score and thus reduces to single-winner SSM.

Use the -s|--score-only option to skip sorted-margins and do only Sequential Monroe Voting.
"""
import argparse
from ballot_tools.csvtoballots import *
import numpy as np
import sorted_margins as sm
import tabulate

myfmt = sm.myfmt

def ssm(ranking,Score,A,cnames,verbose=0):
    """"The basic Score Sorted Margins method (rankings inferred from ratings),
    starting from non-normalized scores and the pairwise array"""
    # Assume the ranking is pre-seeded
    # ranking = Score.argsort()[::-1] # Seed the ranking using Score
    sw = ranking[0]
    sm.sorted_margins(ranking,Score,A.T > A,cnames,verbose=verbose)
    if (verbose > 0) and (len(ranking)>1):
        w  = ranking[0]
        ru = ranking[1]
        w_name = cnames[w]
        ru_name = cnames[ru]
        sw_name = cnames[sw]
        w_score = Score[w]
        sw_score = Score[sw]
        print('[SSM] Winner vs. Runner-up pairwise result: ',
              '{}:{} >= {}:{}'.format(w_name,myfmt(A[w,ru]),
                                      ru_name,myfmt(A[ru,w])))
        if w == sw:
            print('[SSM] Winner has highest score')
        else:
            if sw != ru:
                print('[SSM] Winner vs. highest Scorer, pairwise: ',
                      '{}:{} >= {}:{}'.format(w_name,myfmt(A[w,sw]),
                                              sw_name,myfmt(A[sw,w])))
            print('[SSM] Winner vs. highest Scorer, score: ',
                  '{}:{} <= {}:{}'.format(w_name,myfmt(w_score),
                                          sw_name,myfmt(sw_score)))
    return


# Assumes Hare quota.  For Hagenbach-Bischhoff quota ("Droop"),
# use Numseats minus 1 of the winners, with the last seated winner as
# alternate runner-up.
def ssmpr(ballots, weights, cnames, numseats, verbose=0, score_only=False):
    """Run ssm to elect <numseats> in a PR multiwinner election"""
    numballots, numcands = np.shape(ballots)
    ncands = numcands

    numvotes = weights.sum()
    numvotes_orig = float(numvotes)  # Force a copy

    # Hare quota
    quota = numvotes/numseats

    maxscore = ballots.max()

    cands = np.arange(numcands)

    winners = []

    maxscorep1 = maxscore + 1

    scorerange = np.arange(maxscorep1)

    factor_array = []

    numseatsm1 = numseats - 1

    for seat in range(numseats):

        if verbose>0:
            print("- "*30,"\nStarting count for seat", seat+1)
            print("Number of votes:",myfmt(numvotes))

        # ----------------------------------------------------------------------
        # Tabulation:
        # Round 1: First find scores to determine quota threshold qualifying
        # candidates.
        # ----------------------------------------------------------------------
        S,Q,Score,STotal,qtar = tabulate.scores(ballots,
                                                weights,
                                                cands,
                                                maxscore,
                                                maxscorep1,
                                                quota=quota,
                                                verbose=verbose)

        # Find the score winner,
        # sort qualified candidates by their total score in the top quota
        # of votes approving of that candidate
        inds = np.arange(ncands)
        perm_qc = np.compress(qtar>0,inds) # permuted indices of qualified candidates
        nqcands = len(perm_qc)
        if verbose:
            print("Number of quota-threshold qualifying candidates =",ncands)

        if nqcands == 1:
            # Handle only-one-quota-threshold-qualifying-candidate case:
            permwinner = perm_qc[0]
            winner = cands[permwinner]
            permranking = np.array([0])
        else:
            # Now there is a choice between score_only method and SSM
            if nqcands > 0:
                qcands = cands[perm_qc]
            else:
                if verbose:
                    print("nqcands = 0, resetting to ncands=",ncands)
                nqcands = int(ncands)
                perm_qc = list(inds)
                qcands = list(cands)

            qcnames = cnames[qcands]
            Score_qc = Score[perm_qc]
            permq_rating = list(zip(Score_qc,STotal[perm_qc],qtar[perm_qc]))
            permqranking = np.array(sorted([i for i in range(nqcands)],
                                           key=(lambda c:permq_rating[c]),
                                           reverse=True))

            permqwinner = permqranking[0]

            if verbose > 2:
                print("From top quota score count, sorted ranking:")
                permqtuples = [permq_rating[i] for i in permqranking]
                print("\t" + ",\n\t".join(["{}:{}".format(c,s)
                                          for c, s in zip(cnames[permqranking],permqtuples)]))

            if score_only:
                # don't do Score Sorted Margins, just use the ranking sorted
                # by qualified top score
                permranking = inds[permqranking]
                permwinner = permranking[0]
                
            else:
                # Score Sorted Margins, using top quota score on qualified candidates

                # Retabulation required:
                A = tabulate.pairwise(ballots,
                                      weights,
                                      qcands,
                                      Q[...,perm_qc],
                                      cnames,
                                      maxscore,
                                      maxscorep1,
                                      verbose=verbose)

                # Determine the seat winner using sorted margins elimination:
                ssm(permqranking,Score_qc,A,qcnames,verbose=verbose)
                permqwinner = permqranking[0]
                permranking = inds[permqranking]
                permwinner =  inds[permqwinner]

        winner = cands[permwinner]

        if verbose:
            print("\n-----------\n*** Seat {}: {}\n-----------\n".format(seat+1,cnames[winner]))

        winners += [winner]
        cands = np.compress(cands != winner,cands)
        ncands -= 1

        # Scale weights by proportion of Winner's score that needs to be removed
        factors = np.ones((maxscorep1))
        Sqta = STotal[permwinner]
        Swin = S[...,permwinner]
        v = qtar[permwinner]
        Tqta = Swin[v:].sum()
        winscores = ballots[...,winner]
        vm1 = v - 1
        q = quota

        if verbose:
            print("Quota: ", myfmt(quota))
            print("Winner's quota threshold approval: ", myfmt(Tqta))
            print("Winner's quota threshold rating  : ", v)
            print("Winner's normalized QTA score sum: ", myfmt(Sqta / maxscore))

        if Tqta > quota:
            # Where possible, use score-based scaling
            #
            # It's possible for the quota to be less then the quota threshold approval,
            # but the normalized quota-threshold score to be less than the quota.
            # In such cases, reweight highest scoring seat-winner ballots to zero,
            # then continue to try rescaling the lower scoring ballots proportional to
            # score. The worst case is that all ballots with scores above the quota threshold
            # approval score are reweighted to zero, and ballots at the qta score
            # are the only ones with non-zero reweighting.
            mm = maxscore
            for r in range(maxscore,vm1,-1):
                if (mm * q > Sqta):
                    factors[r] = 0.
                    q -= Swin[r]
                    Sqta -= r * Swin[r]
                    mm = r - 1
                else:
                    factors[r] = 1. - r * q / Sqta

            weights = np.multiply(np.array([factors[s] for s in winscores]), weights)
        else:
            weights = np.where(winscores>0, 0., weights)
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
            print("\tWinner's QTA % before reweighting: {}%".format(myfmt((Tqta/
                                                                           numvotes_orig)*100)))
            print("\tReweighting factor per rating:  ", end="")
            print(", ".join(["{}:{}".format(j,myfmt(f))
                             for j, f in zip(scorerange[-1::-1],
                                             factors[-1::-1])
                             if f < 1.0]))
            print("\t% of vote left:  {}%".format(myfmt((numvotes/
                                                         numvotes_orig) * 100)))
    if (verbose > 1) and (numseats > 1):
        print("- "*30 + "\nReweighting factors for all seat winners:")
        for w, factors in zip(winners,factor_array):
            print(" {} |".format(cnames[w]),
                  ", ".join(["{}:{}".format(j,myfmt(f))
                             for j, f in zip(scorerange[-1:0:-1],
                                             factors[-1:0:-1])
                             if f < 1.0]))

    return(winners)

class SmartDescriptionFormatter(argparse.RawDescriptionHelpFormatter):
  #def _split_lines(self, text, width): # RawTextHelpFormatter, although function name might change depending on Python
  def _fill_text(self, text, width, indent): # RawDescriptionHelpFormatter, although function name might change depending on Python
    #print("splot",text)
    if text.startswith('R|'):
      paragraphs = text[2:].splitlines()
      rebroken = [argparse._textwrap.wrap(tpar, width) for tpar in paragraphs]
      #print(rebroken)
      rebrokenstr = []
      for tlinearr in rebroken:
        if (len(tlinearr) == 0):
          rebrokenstr.append("")
        else:
          for tlinepiece in tlinearr:
            rebrokenstr.append(tlinepiece)
      #print(rebrokenstr)
      return '\n'.join(rebrokenstr) #(argparse._textwrap.wrap(text[2:], width))
    # this is the RawTextHelpFormatter._split_lines
    #return argparse.HelpFormatter._split_lines(self, text, width)
    return argparse.RawDescriptionHelpFormatter._fill_text(self, text, width, indent)

def main():
    from math import log10
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=SmartDescriptionFormatter)

    parser.add_argument("-i", "--inputfile",
                        type=str,
                        required=True,
                        help="REQUIRED: CSV Input file [default: none]")
    parser.add_argument("-m", "--numseats", type=int,
                        default=1,
                        help="Number of seats [default: 1]")
    parser.add_argument("-t", "--filetype", type=str,
                        choices=["score", "rcv"],
                        default="score",
                        help="CSV file type, either 'score' or 'rcv' [default: 'score']")
    parser.add_argument("-s", "--score_only", action='store_true',
                        default=False,
                        help="Score only, no sorted margins (AKA Sequential Monroe) [default: False]")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0,
                        help="Add verbosity [default: 0]")

    args = parser.parse_args()

    ftype={'score':0, 'rcv':1}[args.filetype]

    ballots, weights, cnames = csvtoballots(args.inputfile,ftype=ftype)

    print("- "*30)
    if (args.score_only):
        print("SEQUENTIAL MONROE VOTING")
    else:
        print("SCORE SORTED MARGINS PR VOTING (quota threshold approval)")
    if args.verbose > 3:
        print("- "*30)
        # Figure out the width of the weight field, use it to create the format
        ff = '\t{{:{}d}}:'.format(int(log10(weights.max())) + 1)

        print("Ballots:\n\t{}, {}".format('weight',','.join(cnames)))
        for ballot, w in zip(ballots,weights):
            print(ff.format(w),ballot)

    winners = ssmpr(ballots, weights, cnames, args.numseats,
                    verbose=args.verbose,
                    score_only=args.score_only)
    print("- "*30)

    if args.numseats == 1:
        winfmt = "1 winner"
    else:
        winfmt = "{} winners".format(args.numseats)

    if args.score_only:
        print("\nSMV results, ", end="")
    else:
        print("\nSSMPR results, ", end="")

    print("{}:".format(winfmt),", ".join([cnames[q] for q in winners]))

if __name__ == "__main__":
    main()
