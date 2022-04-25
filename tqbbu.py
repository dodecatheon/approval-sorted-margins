#!/usr/bin/env python
"""
TQBBU:

Top Quota Biased Budgeted Utility

Like Allocated Score, but budget utilities are put into quota-width bins,
in descending order of score, and the average score in each bin is scaled
according to the biasing option.

The candidate with the highest TQBBU wins the seat.

Biasing options are None (== Allocated Score); Arithmetic; Harmonic; Geometric.

For the winner, determine the top quota threshold score (delta), at which a quota
of budget is available at and above score delta.

Then ballots spend their ballots according to the same spending method in 
ranked-MES:

Determine quota threshold score, AKA delta, the score at which a quota of
budget is available for the TQBBU winner.

For all ballots giving the seat winner a score of delta or higher,
determine the price to be exhausted, expending smaller budgets
completely. For remaining ballots, subtract price from budget, equal budget
shares from each ballot.

If TQBBU winner has less than a quota of budget remaining among
supporters, exhaust all supporters' budgets completely
(i.e., delta == 1).
"""
import argparse
from ballot_tools.csvtoballots import *
import numpy as np

import math
log10 = math.log10
isclose = math.isclose
sqrt = math.sqrt
eps = np.finfo(float).eps

import functools

def leq(a, b):
    return a < b or isclose(a, b)

def geq(a, b):
    return a > b or isclose(a, b)

def lessthan(a,b):
    return not geq(a,b)

def greaterthan(a,b):
    return not leq(a,b)

# For use in Python3 tuple sorts:
def mycmp(a, b):
    try:
        if isclose(a[0],b[0]):
            if len(a) > 1:
                # print("len>1, comparing a = ", a, " and b = ", b)
                return mycmp(a[1:],b[1:])
            else:
                # print("len==1, comparing a = ", a, " and b = ", b)
                return 0
        elif a < b:
            return -1
        else:
            return 1
    except:
        # print("scalar, comparing a = ", a, " and b = ", b)
        if isclose(a,b):
            return 0
        elif a < b:
            return -1
        else:
            return 1

# Example:  foo = sorted([list of tuples],reverse=True,key=functools.cmp_to_key(mycmp))
#           bar.sort(reverse=True,key=functools.cmp_to_key(mycmp))

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

def find_support(ballots, weights, budgets, maxscore, numcands):
    mp1 = maxscore + 1
    score_budget = np.zeros((mp1,numcands),dtype=float)
    score_votes  = np.zeros((mp1,numcands),dtype=float)
    utility      = np.zeros((numcands),dtype=float)

    for ballot, w, b in zip(ballots,weights,budgets):
        if b > 0:
            utility += w * b * ballot
            for r in range(maxscore,0,-1):
                rvotes = np.where(ballot==r,w,0)
                score_budget[r] += rvotes * b
                score_votes[r]  += rvotes

    totutil = (weights * budgets * maxscore).sum()

    if totutil > 0:
        utility /= totutil
    else:
        print("Total weighted budget * maxscore == 0")

    return score_budget, score_votes, utility

# NONE
def tqbNone(x,k):
    return x

# Arithmetic
class TqbArithmetic:
    def __init__(self,quota,total):
        self.chunk = quota / total

    def __call__(self,x,k):
        return x * (1. - k*self.chunk)

# Harmonic
def tqbHarmonic(x,k):
    return x / (1+k)

# geometric
class TqbGeometric:
    def __init__(self,inverse_factor):
        self.inverse_factor = inverse_factor

    def __call__(self,x,k):
        return x / self.inverse_factor ** k

def select_one_TQBBU(ballots,
                     weights,
                     budgets,
                     cands,
                     cnames,
                     maxscore,
                     cost,
                     spendable_budget,
                     total_budget,
                     tqb_option=0,
                     verbose=0):

    (numballots, numcands) = np.shape(ballots)

    (score_budget,
     score_votes,
     utility)      = find_support(ballots,
                                  weights,
                                  budgets,
                                  maxscore,
                                  numcands)

    tqb_function = {0:tqbNone,
                    1:TqbArithmetic(cost,total_budget),
                    2:tqbHarmonic,
                    3:TqbGeometric(2),
                    4:TqbGeometric(5/4),
                    5:TqbGeometric(sqrt(2))}[tqb_option]

    if verbose > 1:
        nk = int(total_budget/cost) + 1
        print("Verifying tqb_function, nk =", nk)
        print(", ".join([myfmt(tqb_function(100.,k)) for k in range(nk)]))

    deltas = np.zeros((numcands),dtype=int)
    tqbsums = np.zeros((numcands))
    tqbsum_tuples = []
    for c in cands:
        k = 0
        budgetsum = 0.
        tqbsum    = 0.
        sbc = score_budget[:,c]

        for score in range(maxscore,0,-1):
            lastbsum = float(budgetsum)
            nk = int(((lastbsum + sbc[score]) - k*cost) / cost)
            budgetsum += sbc[score]

            if nk > 0:
                # Keep track of the score at the right side of the first kbox
                # That's the threshold score for spending budgets.
                if k == 0:
                    deltas[c] = score

                # Since there is at least one chunk, take care of the first one:
                tqbsum += tqb_function((cost - (lastbsum % cost)) * score,k)
                k += 1

                # Then do the rest of the chunks
                for kk in range(1,nk):
                    tqbsum += tqb_function(cost * score,k)
                    k += 1

                # Handle the leftover:
                tqbsum += tqb_function((budgetsum - k*cost)*score,k)

            else:
                tqbsum += tqb_function(sbc[score]*score,k)
                
        # If there isn't a full cost-amount of budget for this candidate, deltas[c] == 1
        if k == 0:
            deltas[c] = 1

        tqbsums[c] = float(tqbsum)

    tqbsums /= (weights * budgets * maxscore).sum()

    tqbsum_tuples = sorted([(t,c) for t, c in zip(tqbsums[cands],cands)], reverse=True)

    if verbose > 1:
        for t, c in tqbsum_tuples:
            print ("Candidate", cnames[c], ", tqbsum =", myfmt(t), ", utility =", myfmt(utility[c]) )


    winner_tqbsum, winner = tqbsum_tuples[0]
    if len(tqbsum_tuples) > 1:
        ru_tqbsum, runner_up  = tqbsum_tuples[1]
    else:
        print("Only one candidate")

    # Do cumulative sums:
    # NB: Accumulation is higher to lower
    for score in range(maxscore-1,0,-1):
        score_budget[score] += score_budget[score+1]
        score_votes[score] += score_votes[score+1]

    for delta in range(maxscore,0,-1):
        if leq(cost,score_budget[delta,winner]):
            break

    if delta < 1:
        delta = 1

    if delta != deltas[winner]:
        print("*** Computed delta {} not equal to saved delta {}".format(delta,deltas[winner]))

    delta_budget = score_budget[delta,winner]
    delta_votes  = score_votes[delta,winner]
    
    if leq(delta_budget, cost):
        # All votes at and above score == delta for winner are exhausted
        if verbose > 1:
            print("\t*** select_one_TQBBU: winner budget < cost, delta =", delta)
            print("score_budget = ", delta_budget)
            print("score_votes = ", delta_votes)

        exh_votes  = float(delta_votes)
        exh_budget = float(delta_budget)
        if delta_votes <= 0.:
            if verbose > 0:
                print("delta_votes = 0.")
            price = 10.
        else:
            price = delta_budget / delta_votes

    else:
        budget_tuples = [(b,i)
                        for i, (b, score) in enumerate(zip(budgets,
                                                           ballots[:,winner]))
                        if b > 0 and score >= delta]
        
        min_budget = 10.
        for b, i in budget_tuples:
            if b < min_budget:
                min_budget = float(b)

        # Verify delta_budget:
        bcheck = [b for b, score in zip(budgets,ballots[:,winner])
                    if b > 0 and score >= delta]
        bsumcheck = sum(bcheck)
        blen = len(bcheck)

        if bsumcheck != delta_budget:
            if verbose > 1:
                print("bsumcheck =", bsumcheck, ", delta_budget =", delta_budget, ", blen =", blen)

        remaining_cost = float(cost)
        remaining_votes = float(delta_votes)
        exh_votes  = 0.0
        exh_budget = 0.0
        if remaining_votes > 0.:
            price = remaining_cost / remaining_votes
        else:
            if verbose > 1:
                print("*** remaining_votes <= 0, price set to 10")
            price = 10.

        if leq(price, min_budget):
            if verbose > 1:
                print("Minimum budget on above-threshold ballots is greater than average price, no need for sorting")
        else:
            # Sort budget tuples in place
            budget_tuples.sort()

            # Then find the price for non-exhausted ballots such that cost is spent
            for b, i in budget_tuples:
                w = weights[i]
                wb = w * b
                if leq(price,b):
                    break
                else:
                    if verbose > 2:
                        print("exhausting {} budget".format(wb))
                        print("exhausting {} votes".format(w))
                    remaining_cost  -= wb
                    remaining_votes -= w
                    exh_budget += wb
                    exh_votes += w
                    if (remaining_cost <= 0.) or (remaining_votes <= 0.):
                        if verbose > 2:
                            print("remaining_votes <= 0, last price is sufficient to exhaust all budget")
                            print("remaining_cost = ", remaining_cost)
                            print("cost = ", cost)
                            print("delta_budget = ", delta_budget)
                            print("delta_votes = ", delta_votes)
                        break

                    price = remaining_cost / remaining_votes

    return (winner,
            runner_up,
            price,
            delta,
            delta_budget,
            delta_votes,
            tqbsum_tuples,
            exh_votes,
            exh_budget)

def spend_budget(ballots, weights, budgets, winner, price, delta, spendable_budget, verbose=0):
    # Verify that cost is being spent:
    spent = (np.minimum(np.where(ballots[:,winner]>=delta,price,0.),budgets) * weights).sum()
    if verbose>1:
        print("Spent {}% of budget".format(myfmt(spent/spendable_budget*100)))
    budgets -= np.minimum(np.where(ballots[:,winner]>=delta,price,0.),budgets)
    return spent

def sign(x):
    return 1. if x > 0 else -1.0 if x < 0 else 0

def TQBBU(ballots,
          weights,
          cnames,
          numseats,
          verbose=0,
          qtype=1,
          eliminate_winners=True,
          tqb_option=0):

    """Run TQBBU to elect <numseats> in a PR multiwinner election"""

    numballots, numcands = np.shape(ballots)
    ncands = int(numcands) # force copy
    numvotes = weights.sum()

    # Cost per seat (quota) is Number of votes / Number of seats
    maxscore = ballots.max()
    maxscorep1 = maxscore + 1
    maxscorem1 = maxscore - 1

    # base_budget = 1.0
    total_budget = 1. * numvotes
    cost = numvotes / numseats
    total_budget += cost * qtype

    # Runoff threshold is 0.5%
    recount_threshold = 0.005

    if qtype >= 1:
        # For Droop quota, subtract a small fraction of a vote so we can't
        # pay for an extra seat
        total_budget -= 1./ (64 * numseats)

    budget_per_vote = total_budget / numvotes
    spendable_budget = cost * numseats
    starting_total_budget = 1. * total_budget

    if verbose:
        print("Cost per seat: {}% of spendable budget".format(myfmt(cost/spendable_budget*100)))

    budgets = np.full((numballots), budget_per_vote)
    budget_spent = 0.0
    cands = np.arange(numcands)
    ncands = len(cands)

    winners = []
    runners_up = []
    winprices = []
    deltas = []
    votes = []
    exh_votes = []
    pct_exh_budgets = []
    sorted_tuples_list = []
    pct_budget_spent = []

    for seat in range(numseats):
        if verbose>1:
            print("- "*30,"\nStarting count for seat", seat+1)
            print("Total available budget:  {}%".format(myfmt(total_budget/spendable_budget*100)))

        (winner,
         runner_up,
         maxprice_winner,
         delta,
         delta_budget,
         delta_votes,
         busort,
         exhausted_votes,
         exhausted_budget) = select_one_TQBBU(ballots,
                                              weights,
                                              budgets,
                                              cands,
                                              cnames,
                                              maxscore,
                                              cost,
                                              spendable_budget,
                                              total_budget,
                                              tqb_option=tqb_option,
                                              verbose=verbose)
        vote = delta_votes
        winners += [winner]
        runners_up += [runner_up]
        winprices += [maxprice_winner]
        deltas += [delta]
        votes += [vote]
        exh_votes += [exhausted_votes]
        pct_exh_budgets += [exhausted_budget/spendable_budget * 100]
        sorted_tuples_list += [busort]

        spent = spend_budget(ballots,weights,budgets,winner,maxprice_winner,delta,spendable_budget,verbose=verbose)
        pct_budget_spent += [spent/spendable_budget * 100]

        if eliminate_winners:
            cands = np.compress(cands != winner,cands)
            ncands -= 1

        if verbose:
            if abs(busort[0][0] - busort[1][0]) < recount_threshold:
                print("\n*** For Seat {} --- Runner-up's top-quota-biased budgeted utility within 0.5%, recount required".format(seat+1))
                print("*** Winner {}'s Top-quota-biased budgeted utility: {}%".format(cnames[winner],myfmt(busort[0][0] * 100)))
                print("*** Runner-up {}'s Top-quota-biased budgeted utility: {}%\n".format(cnames[runner_up],myfmt(busort[1][0] * 100)))

        if verbose>1:
            print(("\n-----------\n"
                   "*** Seat {}, delta {}, votes {}: {}, price: {}% of vote; votes exhausted: {}; % budget exhausted: {}%"
                   "\n-----------\n").format(seat+1,
                                             delta,
                                             vote,
                                             cnames[winner],
                                             myfmt(maxprice_winner*100),
                                             exhausted_votes,
                                             myfmt(exhausted_budget/spendable_budget*100)))

        total_budget = (weights*budgets).sum()

        if verbose>1:
            print("- "*30)
            print("After paying for seat:")
            print("\t% of spendable budget spent:  {}%".format(myfmt((starting_total_budget - total_budget)/
                                                                     spendable_budget * 100)))
            print("\t% of total budget remaining:  {}%".format(myfmt(total_budget / spendable_budget * 100)))

    (score_budget,
     score_votes,
     funded_utility) = find_support(ballots,
                                    weights,
                                    budgets,
                                    maxscore,
                                    numcands)
    # Do cumulative sums:
    for score in range(maxscore-1,0,-1):
        score_budget[score] += score_budget[score+1]
        score_votes[score] += score_votes[score+1]

    # find delta
    for delta in range(maxscore,0,-1):
        max_support_max = score_budget[delta,cands].max()
        if leq(cost,max_support_max):
            break

    if verbose > 0:
        print("- "*30)
        print("\t{} winners: ".format(len(winners)), ", ".join([cnames[c] for c in winners]))
        print(("\tRemaining candidates, "
               "delta = {}, "
               "remaining budget {}%:").format(delta,
                                               myfmt(total_budget/spendable_budget * 100)))
        for fu, ms, v, c in sorted(zip(funded_utility[cands],
                                       score_budget[1,cands],
                                       score_votes[1,cands],
                                       cands),
                                   reverse=True):
            print(("\t{}: "
                   "{}% funded utility, "
                   "{}%/cost max support, "
                   "{}% votes").format(cnames[c],
                                       myfmt(fu * total_budget/spendable_budget * 100),
                                       myfmt(ms/cost * 100),
                                       myfmt(v/numvotes*100)))

    if verbose>0:
        print("\n-----------\n")
        for seat, (delta, vote, winner, mpw, exv, pctexb, pctspent) in enumerate(zip(deltas,
                                                                                     votes,
                                                                                     winners,
                                                                                     winprices,
                                                                                     exh_votes,
                                                                                     pct_exh_budgets,
                                                                                     pct_budget_spent)):
            print(("*** Seat {}, delta {}, votes {}: {}, "
                   "price: {}% of vote; "
                   "votes exhausted: {}; "
                   "% budget exhausted: {}%; "
                   "% budget spent: {}%").format(seat+1,
                                                 delta,
                                                 vote,
                                                 cnames[winner],
                                                 myfmt(mpw*100),
                                                 exv,
                                                 myfmt(pctexb),
                                                 myfmt(pctspent)))
        print("-----------")

    return(winners, runners_up)

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
    parser.add_argument("-c", "--approval-cutoff", type=int,
                        default=-1,
                        help="""Use this to toggle between score and approval. 
                        Enter a value >= 0 to count scores over cutoff value as approved
                        [default: -1]""")
    parser.add_argument("-q", "--quota-type",
                        choices=["HARE","DROOP"],
                        default="HARE",
                        help="""Starting budget per ballot.
                        HARE = base budget per ballot = $100 ;
                        DROOP = add base*1/numseats to per-ballot budget corresponding to Droop Quota;
                        [default: HARE]""")
    parser.add_argument('-r','--allow-repeated-winners',
                        action='store_true',
                        default=False,
                        help="Allows candidates to win multiple seats. [default: False]")
    parser.add_argument('-b','--top-quota-biasing',
                        choices=['N', 'NONE',
                                 'A', 'ARITHMETIC',
                                 'H', 'HARMONIC',
                                 'G', 'GEOMETRIC',
                                 '8', '80PCT',
                                 '7', '707PCT'],
                        default="ARITHMETIC",
                        help="""Top quota biasing.
                        N, NONE: Identity function, No biasing, equivalent to Allocated Score *;
                        A, ARITHMETIC: Each quota below top is 1/numseats less than the previous;
                        H, HARMONIC: k-th quota box divided by k;
                        G, GEOMETRIC: k-th quota box divided by 2^k;
                        8, 80PCT: Alternate Geometric -- k-th quota box multiplied by (0.80)^k;
                        7, 707PCT: Alternate Geometric -- k-th quota box multiplied by (1/sqrt(2))^k (i.e., 70.7 pct);
                        [default: ARITHMETIC]""")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0,
                        help="Add verbosity. Repeat to increase level (e.g. -vvv) [default: 0]")
    args = parser.parse_args()

    ballots, weights, cnames = csvtoballots(args.inputfile)

    print("- "*30)

    qtype = {"HARE":0, "DROOP":1}[args.quota_type]

    tqb_option = {"N":0, "A":1, "H":2, "G":3, "8":4, "7":5}[args.top_quota_biasing[0]]
    
    eliminate_winners = not args.allow_repeated_winners

    if args.verbose > 3:
        print("- "*30)
        # Figure out the width of the weight field, use it to create the format
        print("weights.max() =", weights.max(), ", log10(weights.max()) = ", log10(weights.max()))
        ff = '\t{{:{}d}}:'.format(int(log10(weights.max())) + 1)

        print("Ballots:\n\t{}, {}".format('weight',','.join(cnames)))
        for ballot, w in zip(ballots,weights):
            print("\t{}:".format(w),ballot)

    if args.approval_cutoff > -1:
        ballots = np.where(ballots > args.approval_cutoff, 1, 0)
        if args.verbose:
            print("Ballots converted to approval for scores >", args.approval_cutoff)

    (winners,
     runners_up) = TQBBU(ballots, weights, cnames, args.numseats,
                         verbose=args.verbose,
                         qtype=qtype,
                         eliminate_winners=eliminate_winners,
                         tqb_option=tqb_option)
    print("- "*30)

    numwinners = len(winners)
    if numwinners == 1:
        winfmt = "1 winner    "
    else:
        winfmt = "{} winners   ".format(numwinners)

    print("\nTQBBU results:")
    print("\t{}:".format(winfmt),", ".join([cnames[q] for q in winners]))
    print("\t{} runners-up:".format(numwinners),", ".join([cnames[q] for q in runners_up]))

if __name__ == "__main__":
    main()
