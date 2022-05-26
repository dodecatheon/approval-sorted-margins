#!/usr/bin/env python
"""
THS:

Top-Heavy Score (w/Equally Priced subtractive spending)

Like Allocated Score, but budget utilities are put into quota-width bins,
in descending order of score, and the average score in each bin is scaled
according to the biasing option.

The candidate with the highest top-heavy scoresum wins the seat.

Biasing options are None (== Allocated Score); Arithmetic; Harmonic; Geometric.

For the winner, determine the top quota threshold score (delta), at which a quota
of budget is available at and above score delta.

Then ballots spend their ballots according to the same spending method in 
ranked-MES:

Determine quota threshold score, AKA delta, the score at which a quota of
budget is available for the THS winner.

For all ballots giving the seat winner a score of delta or higher,
determine the price to be exhausted, expending smaller budgets
completely. For remaining ballots, subtract price from budget, equal budget
shares from each ballot.

If THS winner has less than a quota of budget remaining among
supporters, exhaust all supporters' budgets completely
(i.e., delta == 1).
"""
import argparse
from ballot_tools.csvtoballots import *
import numpy as np
import scipy.special as sp
from itertools import combinations
from collections import defaultdict

import math
log10 = math.log10
log = math.log
digamma = sp.digamma
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
def thsNone(x,k):
    return x

# Arithmetic
class thsArithmetic:
    def __init__(self,quota,total):
        self.chunk = quota / total

    def __call__(self,x,k):
        return x * (1. - k*self.chunk)

# Harmonic
def thsHarmonic(x,k):
    return x / (1+k)

# geometric
class thsGeometric:
    def __init__(self,inverse_factor):
        self.inverse_factor = inverse_factor

    def __call__(self,x,k):
        return x / self.inverse_factor ** k

def select_one_THS(ballots,
                   weights,
                   budgets,
                   cands,
                   cnames,
                   maxscore,
                   cost,
                   spendable_budget,
                   total_budget,
                   ths_option=0,
                   approval_cutoff=-1,
                   verbose=0):

    (numballots, numcands) = np.shape(ballots)

    (score_budget,
     score_votes,
     utility)      = find_support(ballots,
                                  weights,
                                  budgets,
                                  maxscore,
                                  numcands)

    ths_function = {0:thsNone,
                    1:thsGeometric(2),
                    2:thsGeometric(3/2),
                    3:thsGeometric(4/3),
                    4:thsGeometric(5/4),
                    5:thsGeometric(6/5),
                    6:thsArithmetic(cost,total_budget),
                    7:thsHarmonic}[ths_option]

    if verbose > 1:
        nk = int(total_budget/cost) + 1
        print("Verifying ths_function, nk =", nk)
        print(", ".join([myfmt(ths_function(100.,k)) for k in range(nk)]))

    deltas = np.zeros((numcands),dtype=int)
    thssums = np.zeros((numcands))
    thssum_tuples = []
    for c in cands:
        k = 0
        budgetsum = 0.
        thssum    = 0.
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
                thssum += ths_function((cost - (lastbsum % cost)) * score,k)
                k += 1

                # Then do the rest of the chunks
                for kk in range(1,nk):
                    thssum += ths_function(cost * score,k)
                    k += 1

                # Handle the leftover:
                thssum += ths_function((budgetsum - k*cost)*score,k)

            else:
                thssum += ths_function(sbc[score]*score,k)
                
        # If there isn't a full cost-amount of budget for this candidate, deltas[c] == 1
        if k == 0:
            deltas[c] = 1

        thssums[c] = float(thssum)

    thssums /= (weights * budgets * maxscore).sum()

    thssum_tuples = sorted([(t,c) for t, c in zip(thssums[cands],cands)], reverse=True)

    if verbose > 1:
        for t, c in thssum_tuples:
            print ("Candidate", cnames[c], ", thssum =", myfmt(t), ", utility =", myfmt(utility[c]) )


    winner_thssum, winner = thssum_tuples[0]
    if len(thssum_tuples) > 1:
        ru_thssum, runner_up  = thssum_tuples[1]
    else:
        print("Only one candidate")

    # Do cumulative sums:
    # NB: Accumulation is higher to lower
    for score in range(maxscore-1,0,-1):
        score_budget[score] += score_budget[score+1]
        score_votes[score] += score_votes[score+1]

    #
    # Compute delta (quota-threshold approval), with a minimum value of 1,
    # if we haven't forced a hardcoded approval cutoff
    #
    if approval_cutoff < 0:
        for delta in range(maxscore,0,-1):
            if leq(cost,score_budget[delta,winner]):
                break

        if delta < 1:
            delta = 1

        if delta != deltas[winner]:
            print("\n\t*** Computed delta {} not equal to saved delta {}".format(delta,deltas[winner]))
    else:
        delta = approval_cutoff + 1
        if verbose > 0:
            print("\n\t*** Forcing approval cutoff, minimum approved score = {}".format(delta))

    delta_budget = score_budget[delta,winner]
    delta_votes  = score_votes[delta,winner]

    remaining_cost = 1.0 * cost
    remaining_votes = 1.0 * delta_votes
    if remaining_votes > 0.:
        price = remaining_cost / remaining_votes
    else:
        price = 1.
        if verbose > 1:
            print("\n\t*** remaining_votes <= 0, price set to 1.0\n")

    exh_votes = 0.0
    exh_budget = 0.0
    exh_ballots = 0
    delta_ballots = 0
    
    if leq(delta_budget, cost):
        # All votes at and above score == delta for winner are exhausted
        if verbose > 0:
            print("\n\t*** select_one_THS: winner budget < cost, delta =", delta, ", all delta+ ballots exhausted")
            print("\tExhausted budget = ", delta_budget)
            print("\tExhausted votes = ", delta_votes)

        exh_votes  += delta_votes
        exh_budget += delta_budget
        price = np.where(ballots[:,winner]>=delta,budgets,0.0).max()
        if price == 0.0:
            price = 1.0
        if verbose > 0:
            print("\n\tPrice for all exhausted ballots = max budget, {}% of a vote\n".format(myfmt(price*100)))

    else:

        # Find min_budget and find total ballots for each budget value.
        budget_counts = defaultdict(int)
        min_budget = 10.0
        delta_ballots = 0
        for score, w, b in zip(ballots[:,winner],weights,budgets):
            if (b > 0.0) and (score >= delta):
                if b <= price:
                    exh_budget += w*b
                    exh_votes += w
                    exh_ballots += 1
                else:
                    if b < min_budget:
                        min_budget = float(b)
                    budget_counts[b] += w
                    delta_ballots += 1

        if verbose > 0:
            print("\n\tInitial price check filtered {} ballots, {} votes out of {} ballots, {} votes".format(exh_ballots,
                                                                                                             exh_votes,
                                                                                                             exh_ballots + delta_ballots,
                                                                                                             delta_votes))
        remaining_votes -= exh_votes
        remaining_cost -= exh_budget
        if remaining_votes > 0.:
            price = remaining_cost / remaining_votes
        else:
            price = np.where(ballots[:,winner]>=delta,budgets,0.0).max()
            if price == 0.0:
                price = 1.0

        if leq(price, min_budget):
            if verbose > 1:
                print("\n\tMinimum budget on above-threshold ballots is greater than average price, no need for sorting\n")
        else:
            sorted_bctuples = sorted([pair for pair in budget_counts.items()])

            if verbose:
                print("Dict method reduces sorting length from {} to {}".format(delta_ballots,
                                                                                len(sorted_bctuples)))

            # Find the price for non-exhausted ballots such that cost is spent
            for b, w in sorted_bctuples:
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
            thssum_tuples,
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

def THS(ballots,
        weights,
        cnames,
        numseats,
        verbose=0,
        qtype=1,
        eliminate_winners=True,
        approval_cutoff=-1,
        ths_option=0):

    """Run THS to elect <numseats> in a PR multiwinner election"""

    numballots, numcands = np.shape(ballots)
    ncands = int(numcands) # force copy
    numvotes = weights.sum()

    # Cost per seat (quota) is Number of votes / Number of seats
    maxscore = ballots.max()
    maxscorep1 = maxscore + 1
    maxscorem1 = maxscore - 1

    # base_budget = 1.0
    total_budget = 1. * numvotes
    cost = numvotes / (numseats + qtype)

    # Runoff threshold is 0.5%
    recount_threshold = 0.005

    if qtype >= 1:
        # For Droop quota, add a small fraction of a vote so we can't
        # pay for an extra seat
        cost += 1./ (16 * numseats)

    budget_per_vote = 1.0
    spendable_budget = 1.0 * total_budget

    if verbose:
        print("Cost per seat (i.e. quota): {}% of spendable budget".format(myfmt(cost/spendable_budget*100)))

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
         exhausted_budget) = select_one_THS(ballots,
                                              weights,
                                              budgets,
                                              cands,
                                              cnames,
                                              maxscore,
                                              cost,
                                              spendable_budget,
                                              total_budget,
                                              ths_option=ths_option,
                                              approval_cutoff=approval_cutoff,
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
                print("\n*** For Seat {} --- Runner-up's top-heavy utility within 0.5%, recount required".format(seat+1))
                print("*** Winner {}'s top-heavy utility: {}%".format(cnames[winner],myfmt(busort[0][0] * 100)))
                print("*** Runner-up {}'s top-heavy utility: {}%\n".format(cnames[runner_up],myfmt(busort[1][0] * 100)))

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
            print("\t% of spendable budget spent:  {}%".format(myfmt((spendable_budget - total_budget)/
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
        print(("\n\tRemaining candidates, "
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
        print("\n-----------\n")
        if verbose > 3:
            mp1 = maxscore + 1
            util = np.arange(mp1) / maxscore
            dgshift = 1 - digamma(2.)

            all_win_tups = [tup for tup in combinations(range(numcands),numseats)]
            num_sets = len(all_win_tups)
            all_num_winners = np.zeros((num_sets))
            all_ge_one_winner = np.zeros((num_sets))
            all_harmonic_util = np.zeros((num_sets))
            all_log_util = np.zeros((num_sets))

            wintuple = tuple(winners)

            for i, win_set in enumerate(all_win_tups):
                win_list = list(win_set)
                anw = 0.0
                alo = 0.0
                ahu = 0.0
                alu = 0.0
                for ballot, w in zip(ballots,weights):
                    b = ballot[win_list]
                    if np.any(b>0):
                        anw += w * b.compress(b>0).size
                        alo += w
                        usum = util[b].sum()
                        ahu += w*(digamma(1+usum)+dgshift)
                        alu += w*log(0.5+usum)
                all_num_winners[i] = anw / numvotes
                all_ge_one_winner[i] = alo / numvotes
                all_harmonic_util[i] = ahu / numvotes
                all_log_util[i] = alu / numvotes

            i_win = all_win_tups.index(tuple(sorted(winners)))

            i_anw = all_num_winners.argmax()
            i_alo = all_ge_one_winner.argmax()
            i_ahu = all_harmonic_util.argmax()
            i_alu = all_log_util.argmax()

            avg_num_winners = all_num_winners[i_win]
            at_least_one_w = all_ge_one_winner[i_win]
            avg_harmonic_utility = all_harmonic_util[i_win]
            avg_log_utility = all_log_util[i_win]

            print("\tAverage number of winners per ballot  = {}".format(myfmt(avg_num_winners)),
                  ", {}% of max with winners ({})".format(myfmt(avg_num_winners/all_num_winners[i_anw]*100),
                                                          ", ".join(cnames[list(all_win_tups[i_anw])])))
            print("\t# of ballots with at least one winner = {}%".format(myfmt(at_least_one_w*100)),
                  ", {}% of max with winners ({})".format(myfmt(at_least_one_w/all_ge_one_winner[i_alo]*100),
                                                          ", ".join(cnames[list(all_win_tups[i_alo])])))
            print("\tAverage harmonic utility =", myfmt(avg_harmonic_utility),
                  ", {}% of max with winners ({})".format(myfmt(avg_harmonic_utility/all_harmonic_util[i_ahu]*100),
                                                          ", ".join(cnames[list(all_win_tups[i_ahu])])))
            print("\tAverage log utility =", myfmt(avg_log_utility),
                  ", {}% of max with winners ({})".format(myfmt(avg_log_utility/all_log_util[i_alu]*100),
                                                          ", ".join(cnames[list(all_win_tups[i_alu])])))
            print("\n\tMax avg#winners set ({}):".format(", ".join(cnames[list(all_win_tups[i_anw])])),
                  "\n\t\t {}, {}%, {}, {}".format(myfmt(all_num_winners[i_anw]),
                                                  myfmt(all_ge_one_winner[i_anw]*100),
                                                  myfmt(all_harmonic_util[i_anw]),
                                                  myfmt(all_log_util[i_anw])))
            print("\n\tMax at least one winner set ({}):".format(", ".join(cnames[list(all_win_tups[i_alo])])),
                  "\n\t\t {}, {}%, {}, {}".format(myfmt(all_num_winners[i_alo]),
                                                  myfmt(all_ge_one_winner[i_alo]*100),
                                                  myfmt(all_harmonic_util[i_alo]),
                                                  myfmt(all_log_util[i_alo])))
            print("\n\tMax avg harmonic utility set ({}):".format(", ".join(cnames[list(all_win_tups[i_ahu])])),
                  "\n\t\t {}, {}%, {}, {}".format(myfmt(all_num_winners[i_ahu]),
                                                  myfmt(all_ge_one_winner[i_ahu]*100),
                                                  myfmt(all_harmonic_util[i_ahu]),
                                                  myfmt(all_log_util[i_ahu])))
            print("\n\tMax avg log utility set ({}):".format(", ".join(cnames[list(all_win_tups[i_alu])])),
                  "\n\t\t {}, {}%, {}, {}".format(myfmt(all_num_winners[i_alu]),
                                                  myfmt(all_ge_one_winner[i_alu]*100),
                                                  myfmt(all_harmonic_util[i_alu]),
                                                  myfmt(all_log_util[i_alu])))
            print("\n-----------\n")

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
                        choices=["HARE","DROOP","0", "1"],
                        default="HARE",
                        help="""Starting budget per ballot.
                        0, HARE = Cost per seat (quota) = numseats / numvotes ;
                        1, DROOP = Cost per seat (quota) = (numseats / (numvotes + 1)) plus a small fraction of a vote. 
                        [default: HARE]""")
    parser.add_argument('-r','--allow-repeated-winners',
                        action='store_true',
                        default=False,
                        help="Allows candidates to win multiple seats. [default: False]")
    parser.add_argument('-t','--top-heavy-method',
                        choices=['N', 'NONE',
                                 '1', '1GEOMETRIC',
                                 '2', '2GEOMETRIC',
                                 '3', '3GEOMETRIC',
                                 '4', '4GEOMETRIC',
                                 '5', '5GEOMETRIC',
                                 'A', 'ARITHMETIC',
                                 'H', 'HARMONIC'],
                        default="4GEOMETRIC",
                        help="""Top-heavy method.
                        N, NONE: Identity function, No biasing, equivalent to Allocated Score *;
                        A, ARITHMETIC: Each quota below top is 1/numseats less than the previous;
                        H, HARMONIC: k-th quota box divided by k;
                        #, #GEOMETRIC: Geometric scaling. k-th quota box is 1/(1 + 1/#) times the (k-1)-st quota box;
                        1, 1GEOMETRIC: 50pct; 2: 66.6pct; 3: 75pct; 4: 80pct; 5: 83.3pct;
                        [default: 4GEOM = 80pct geometric reduction]""")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0,
                        help="Add verbosity. Repeat to increase level (e.g. -vvv) [default: 0]")
    args = parser.parse_args()

    ballots, weights, cnames = csvtoballots(args.inputfile)

    print("- "*30)

    qtype = {"HARE":0, "DROOP":1, "0":0, "1":1}[args.quota_type]

    ths_option = {"N":0, "1":1, "2":2, "3":3, "4":4, "5":5, "A":6, "H":7 }[args.top_heavy_method[0]]
    
    eliminate_winners = not args.allow_repeated_winners

    if args.verbose > 3:
        print("- "*30)
        # Figure out the width of the weight field, use it to create the format
        print("weights.max() =", weights.max(), ", log10(weights.max()) = ", log10(weights.max()))
        ff = '\t{{:{}d}}:'.format(int(log10(weights.max())) + 1)

        print("Ballots:\n\t{}, {}".format('weight',','.join(cnames)))
        for ballot, w in zip(ballots,weights):
            print("\t{}:".format(w),ballot)

    (winners,
     runners_up) = THS(ballots, weights, cnames, args.numseats,
                       verbose=args.verbose,
                       qtype=qtype,
                       eliminate_winners=eliminate_winners,
                       approval_cutoff=args.approval_cutoff,
                       ths_option=ths_option)
    print("- "*30)

    numwinners = len(winners)
    if numwinners == 1:
        winfmt = "1 winner    "
    else:
        winfmt = "{} winners   ".format(numwinners)

    print("\nTHS results:")
    print("\t{}:".format(winfmt),", ".join([cnames[q] for q in winners]))
    print("\t{} runners-up:".format(numwinners),", ".join([cnames[q] for q in runners_up]))

if __name__ == "__main__":
    main()
