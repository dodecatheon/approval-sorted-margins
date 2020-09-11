# Approval Sorted Margins
## A Condorcet-completion method with explicit approval cutoff

Approval Sorted Margins was introduced by Forest Simmons as a symmetric
modification to Definitive Majority Choice (AKA Ranked Approval Voting).

https://electowiki.org/wiki/Approval_Sorted_Margins

Code for this, compared with other approval cutoff schemes, may be found
in

    asm.py

A number of cases can be found in the rating_examples submodule.

## Score Sorted Margins as part of a PR method (based on Sequential Monroe Voting)
A variant that applies Sorted Margins on a seed of Top Quota Scores (see Sequential
Monroe Voting), instead of Approval with cutoff, may be used in a form of Quota-based Threshold
Approval PR voting. This code can be found in 

    ssmpr.py

which will run Score Sorted Margins as a multiwinner election.  By default, it is
run with Hare quota. For Droop quota and some measure of Droop Proportionality,
run it with one more seat than desired, then use the last seat as a runner-up.

Example usage (add -v for verbose level 1, -vv for verbose level 2, etc.

	ssmpr.py -vv -i rating_examples/rlg_approval_cutoff_3.csv

Multiwinner, 9 seats:

	ssmpr.py -m 9 -vv -i rating_examples/june2011.csv

RCV election in Minnesota, 2013:

	ssmpr.py -t rcv -vv -i rating_examples/actual-mayoral-election.csv

Since the seeded ranking is based on the same top quota score as
in Sequential Monroe Voting, a version of SMV can be run using
the -s option. Run ssmpr.py with the -h option for more details
on how the top quota score is determined.

Comparing SSMPR and SMV, similar results are found, indicating
that in the multiwinner case, it may be more efficient to simply
use SMV.

### SSMPR reweighting
Note that an essential aspect of a sequential quota-based PR method
is how winner-voting ballots are reweighted.

In Sequential Monroe Voting, all ballots with scores above the Quota
Threshold Rating, score > v, are reweighted to zero, and the remainder
of the quota is removed from ballots voting exactly at the threshold
rating score == v.  Interestingly enough, the resulting weighting factors
as a function of ballot score are exactly the complement of how weighted
ballots are accumulated into the pairwise array. Therefore, this is the
default mode of reweighting for SSMPR.

In Single Transferable Vote, all ballots after elimination that vote
for the seat winner are reweighted using

    Factor = 1 - Quota / Total-winning-votes

This can be adapted to ssmpr.py by using the total weighted sum of ballots
at and above the quota threshold rating (QTA) as the Total.  Select this option
using the arguments '-r STV'.


    QTAScoreSum = Sum(v*S[v] + (v+1)*S[v+1] + ... + maxscore * S[maxscore])
        for v = quota threshold approval rating

    Factor[r] = 1 - r * Quota / QTAScoreSum

However, it is possible for Quota / Quota-Threshold-Approval to be
less than one while maxscore * Quota / QTAScoreSum is greater than one.

In such cases, redefine QTAScoreSum for maximum scores >= maxscore, as

    QTAScoreSum[mm, v] = Sum(S[maxscore]*mm + S[maxscore-1]*(mm-1) + ... 
                                 + S[v] * (v + mm-maxscore)

We then successively increase the score range mm until

    newmax * quota <= QTAScoreSum[newmax,v]

and then, for scores 
 
    Factor[r] = 1 - (r + newmax-maxscore) * Quota / QTAScoreSum[newmax,vcomplement]
    
It can be seen that as mm goes to infinity, the rescaling factor approaches
the 'STV' reweighting factor asymptotically.

Select this option using the argument '-r scaled'.

When the quota-threshold-approval rating is zero, the winner did not achieve
a full quota of the vote, and all ballots giving non-zero score to that seat
winner are reweighted to zero.

## winnow.py, a primary-style election
Finally, both ASM and SSM can be used in a primary-style winnowing election.
Code for this can be found in 'winnow.py'.

For a winnowing election, vote using score ballots. Any non-zero score is taken
as approval. A voter can optionally specify a Preference cutoff above the default
preference cutoff score of zero. A score above the cutoff is counted as a Preference
vote.

Repeat until no candidates remain with approval greater than a threshold (default 1%).

* Tabulate preference, approval, score and pairwise votes on ballots that still have weight > 0. Total approval over these ballots is called exclusive approval, since it excludes any approval from previously used-up ballots.
* Remove any candidate with exclusive approval below the approval threshold.
* A winner is found, using one of the methods Preference Approval Sorted Margins, Preference, Score, STAR, or SSM.

Exhausting each winner's ballots completely removes any potential for a pushover
strategy.

Other than STAR, each of the possible methods is clone resistant. Advancing only
one winner in the remaining ballots removes clones.

The resulting set of advanced candidates gives a reasonable selection among the
best representatives of each faction, enabling use of a non-clone-resistant method
for the general election, while also enabling a fully open primary.

Political parties have traditionally opposed open primaries because of the potential
for people to strategically nominate the weakest opponent of their opposition. The
'winnow' method might still allow this to the same degree as current closed primaries,
but strategic voters from party X would have to gamble that their favored candidate would still win
an advancement round without their support, and with other voters from parties Y and Z
similarly able to support the weaker candidates from party X, the game theoretic
equilibrium will tend toward sincere preferences.
