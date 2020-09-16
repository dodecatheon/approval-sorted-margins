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

def ssm(ranking,Score,A,cnames,verbose=0,Tied=None):
    """"The basic Score Sorted Margins method (rankings inferred from ratings),
    starting from non-normalized scores and the pairwise array"""
    # Assume the ranking is pre-seeded
    # ranking = Score.argsort()[::-1] # Seed the ranking using Score
    sw = ranking[0]
    STAR_winner = ranking[0]
    STAR_runnerup = ranking[1]
    if len(ranking) > 1:
        sru = ranking[1]
        if A[sru,sw] > A[sw,sru]:
            STAR_winner, STAR_runnerup = STAR_runnerup, STAR_winner

    sm.sorted_margins(ranking,Score,A.T > A,cnames,verbose=verbose)
    if (verbose > 0) and (len(ranking)>1):
        w  = ranking[0]
        ru = ranking[1]
        w_name = cnames[w]
        ru_name = cnames[ru]
        sw_name = cnames[sw]
        sru_name = cnames[sru]
        STARw_name = cnames[STAR_winner]
        STARru_name = cnames[STAR_runnerup]
        w_score = Score[w]
        sw_score = Score[sw]
        STARw_score = Score[STAR_winner]

        ssm_tied = ""
        ssm_STAR_tied = ""
        score_tied = ""
        ssm_score_tied = ""
        if (np.shape(Tied)):
            ssm_tied = ", with {} votes tied & approved".format(myfmt(Tied[w,ru]))
            ssm_STAR_tied = ", with {} votes tied & approved".format(myfmt(Tied[w,STAR_winner]))
            score_tied = ", with {} votes tied & approved".format(myfmt(Tied[sw,sru]))
            ssm_score_tied = ", with {} votes tied & approved".format(myfmt(Tied[w,sw]))

        print(('[SSM] Winner vs. SSM-Runner-up pairwise result:  ' + 
               '{} @ {} >= {} @ {}' +
               ssm_tied).format(w_name,myfmt(A[w,ru]),
                               ru_name,myfmt(A[ru,w])))
        if STAR_winner != sw:
            print(('STAR Winner defeats Score winner, pairwise:  ' +
                  '{} @ {} >= {} @ {}' +
                  score_tied).format(STARw_name,myfmt(A[STAR_winner,sw]),
                                     sw_name,myfmt(A[sw,STAR_winner])))
        else:
            print('STAR winner == Score winner')
            print(('Score Winner vs. Score runner-up, pairwise:  ' +
                   '{} @ {} >= {} @ {}' +
                   score_tied).format(sw_name,myfmt(A[sw,sru]),
                                      sru_name,myfmt(A[sru,sw])))

        if w == sw:
            if w == STAR_winner:
                print('[SSM] Winner is both Score winner and STAR winner')
            else:
                print('[SSM] Winner is Score winner, but not STAR winner')
        else:
            if w == STAR_winner:
                print('[SSM] Winner is not Score winner, but is STAR winner')

            if sw != ru:
                print(('[SSM] Winner vs. Score winner, pairwise: ' +
                       '{} @ {} >= {} @ {}' +
                       ssm_score_tied).format(w_name,myfmt(A[w,sw]),
                                              sw_name,myfmt(A[sw,w])))
            print(('[SSM] Winner vs. Score Winner, score: ' +
                   '{} @ {} <= {} @ {}' +
                   ssm_score_tied).format(w_name,myfmt(w_score),
                                          sw_name,myfmt(sw_score)))
        if sw != STAR_winner and w != STAR_winner:
            print(('[SSM] Winner vs. STAR winner, pairwise:  ' +
                   '{} @ {} >= {} @ {}' +
                   ssm_STAR_tied).format(w_name,myfmt(A[w,STAR_winner]),
                                         STARw_name,myfmt(A[STAR_winner,w])))
    return STAR_winner


# Assumes Hare quota.  For Hagenbach-Bischhoff quota ("Droop"),
# use Numseats minus 1 of the winners, with the last seated winner as
# alternate runner-up.
def ssmpr(ballots, weights, cnames, numseats, reweighting=0,verbose=0, method=0):
    """Run ssm to elect <numseats> in a PR multiwinner election"""
    numballots, numcands = np.shape(ballots)
    ncands = int(numcands) # force copy

    if (numseats > ncands):
        print("*** WARNING, # seats > # candidates! ***")
        print("Resetting # seats to", ncands)
        numseats = int(numcands)

    numvotes = weights.sum()
    numvotes_orig = float(numvotes)  # Force a copy

    # Hare quota
    quota = numvotes/numseats
    droopquota = numvotes / (numseats + 1)

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
            # Now there is a choice between SMV and SSM
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
            QTA = [S[qtar[c]:,c].sum() for c in perm_qc]
            permq_rating = list(zip(Score_qc,STotal[perm_qc],qtar[perm_qc],QTA))
            permqranking = np.array(sorted(np.arange(nqcands),
                                           key=(lambda c:permq_rating[c]),
                                           reverse=True))
            permqwinner = permqranking[0]

            if verbose > 2:
                print("From top quota score count, sorted ranking:")
                print("\tCandidate: (TopQuotaScore,QTA_Score,QTA_Rating,QTA)")
                permqtuples = [permq_rating[i] for i in permqranking]
                print("\t" + ",\n\t".join(["{}:{}".format(c,s)
                                          for c, s in zip(cnames[permqranking],permqtuples)]))

            permranking = inds[permqranking]

            if method==1:                    # SMV
                # don't do Score Sorted Margins, just use the ranking sorted
                # by qualified top score
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
                                      verbose=verbose)

                # Determine the seat winner using sorted margins elimination:
                STAR_winner = ssm(permqranking,permq_rating,A,qcnames,verbose=verbose)
                # STAR_winner = ssm(permqranking,Score_qc,A,qcnames,verbose=verbose)
                if (method == 2):
                    permqwinner = int(STAR_winner)
                else:
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
        vp1 = v + 1
        q = float(quota) # copy

        if verbose:
            print("Quota: ", myfmt(quota))
            print("Winner's quota threshold approval: ", myfmt(Tqta))
            print("Winner's quota threshold rating  : ", v)
            print("Winner's normalized QTA score sum: ", myfmt(Sqta / maxscore))

        if Tqta <= quota:
            weights = np.where(winscores>0, 0., weights)
            factors = np.zeros((maxscorep1))
        else:
            # Variations on reweighting:
            if (v == maxscore):
                factors[v] = 1.0 - q / Swin[v]
                if verbose > 0:
                    print("QTA rating == maxscore ({}), so all reweighting algorithms are the same.".format(maxscore))
            elif (reweighting == 2):        # STV
                # For Single Transferable Vote, or if the quota threshold rating
                # is maxscore, just rescale the top-rating-scored ballots to remove
                # an entire quota
                factors[v:] = 1.0 - q / Tqta
                if verbose > 0:
                    print("STV reweighting")
            elif reweighting == 1:          # SMV
                # In classical Sequential Monroe Voting, all ballots scoring the
                # winner /above/ the quota-threshold-rating are reweighted to
                # zero. Any weight still remaining is rescaled to be removed from
                # ballots on the quota-threshold-rating cusp

                if True:
                    # James Quinn refinement: exhaust only a droopquota for each seat
                    # instead of Hare quota, while still using Hare quota for quota threshold rating, etc.
                    q = float(droopquota)   # copy
                    Twin = 0.
                    for r in range(maxscore,0,-1):
                        Twin += Swin[r]
                        if Twin < droopquota:
                            factors[r] = 0.0
                            q -= Swin[r]
                        elif Twin >= droopquota:
                            factors[r] = 1 - q / Swin[r]
                            break
                else:                                       # Original SMV factors:
                    q -= Swin[vp1:].sum()
                    factors[v] = 1.0 - q / Swin[v]
                    factors[vp1:] = 0.0

                if verbose > 0:
                    print("SMV reweighting")
            else:                           # Scaled
                # Scaled reweighting is a compromise between STV and SMV:
                # The qta score sum rating range is adjusted upward until 
                # the quota-threshold score sum is larger than the quota
                rr = scorerange[v:] + 0
                ss = (Swin[v:] * rr).sum()
                while ( (rr[-1] * q) > ss ):
                    rr += 1
                    ss = (Swin[v:] * rr).sum()
                factors[v:] = 1.0 - rr * q / ss
                if verbose > 0:
                    print("Scaled reweighting")
                    if (rr[-1] > maxscore):
                        print(("\tQTA+ scores adjusted "
                               "from {}:{} to {}:{}").format(int(scorerange[v]),
                                                             int(scorerange[-1]),
                                                             int(rr[0]),
                                                             int(rr[-1])))

            # for each ballot, rescale its weight by the factor corresponding to
            # the winning candidate's score on that ballot:
            weights = np.multiply(np.array([factors[s] for s in winscores]), weights)

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
            print("\tQfactors:  ", end="")
            Qfactors=1. - Q[...,permwinner]
            print(", ".join(["{}:{}".format(j,myfmt(f))
                             for j, f in zip(scorerange[-1::-1],
                                             Qfactors[-1::-1])
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
    parser.add_argument("-r", "--reweighting", type=str,
                        choices=["SMV", "STV", "Scaled"],
                        default="SMV",
                        help="""Reweighting algorithm choice:
                        'SMV' = sequential monroe voting style, any scores above the
                        quota threshold approval (QTA) rating are reweighted to zero,
                        with scores exactly at the QTA rating rescaled to finish removing
                        one quota;
                        'STV' = Single Transferable Vote style, all scores at and above
                        the QTA rating use the same factor; or
                        'Scaled', scores are optimally reweighted proportional to adjusted
                        score (see README for details), a compromise between SMV and STV
                        styles. [default: 'SMV']""")
    parser.add_argument("-s", "--select_method",
                        choices=["SSM", "SMV", "STAR"],
                        default="SSM",
                        help="""Select method for seat winner: SSM (Score Sorted Margins, Condorcet), SMV(Sequential
                        Monroe Voting, top score), or STAR (Score Then Automatic Runoff, pairwise between top two
                        quota threshold score winners [default: SSM]""")
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

    print("- "*30)

    method = {"SSM":0, "SMV":1, "STAR":2}[args.select_method]

    method_description = ["SCORE SORTED MARGINS PR",
                          "SEQUENTIAL MONROE VOTING",
                          "STAR = Score Then Automatic Runoff, with SMV weighting"]

    print(method_description[method])

    if args.verbose > 0:
        print("\tWith reweighting method =", args.reweighting)

    reweighting = {"Scaled":0,"SMV":1,"STV":2}[args.reweighting]

    if args.verbose > 3:
        print("- "*30)
        # Figure out the width of the weight field, use it to create the format
        ff = '\t{{:{}d}}:'.format(int(log10(weights.max())) + 1)

        print("Ballots:\n\t{}, {}".format('weight',','.join(cnames)))
        for ballot, w in zip(ballots,weights):
            print(ff.format(w),ballot)

    winners = ssmpr(ballots, weights, cnames, args.numseats,
                    reweighting=reweighting,
                    verbose=args.verbose,
                    method=method)
    print("- "*30)

    if args.numseats == 1:
        winfmt = "1 winner"
    else:
        winfmt = "{} winners".format(args.numseats)

    method_descriptor = {0:"SSMPR", 1:"SMV", 2:"STAR"}[method]
    print("\n{} results, ".format(method_descriptor), end="")
    print("{}:".format(winfmt),", ".join([cnames[q] for q in winners]))

if __name__ == "__main__":
    main()
