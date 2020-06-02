# Approval Sorted Margins
## A Condorcet-completion method with explicit approval cutoff

Approval Sorted Margins was introduced by Forest Simmons as a symmetric
modification to Definitive Majority Choice (AKA Ranked Approval Voting).

https://electowiki.org/wiki/Approval_Sorted_Margins

Code for this, compared with other approval cutoff schemes, may be found
in

    asm.py

A variant that applies Sorted Margins on a seed of Top Quota Scores, instead
of Approval with cutoff, may be used in a form of Quota-based Threshold
Approval PR voting. This code can be found in 

    ssmpr.py

which will run Score Sorted Margins as a multiwinner election.  By default, it is
run with Hare quota. For Droop quota and some measure of Droop Proportionality,
run it with one more seat than desired, then use the last seat as a runner-up.

Example usage (add -v for verbose level 1, -vv for verbose level 2, etc.

	ssmpr.py -vv -i examples/rlg_approval_cutoff_3.csv

Multiwinner, 9 seats:

	ssmpr.py -m 9 -vv -i examples/june2011.csv

RCV election in Minnesota, 2013:

	ssmpr.py -t rcv -vv -i examples/actual-mayoral-election.csv

Since the seeded ranking is based on the same top quota score as
in Sequential Monroe Voting, a version of SMV can be run using
the -s option. Run ssmpr.py with the -h option for more details
on how the top quota score is determined.

Comparing SSMPR and SMV, similar results are found, indicating
that in the multiwinner case, it may be more efficient to simply
use SMV.

Note that an essential aspect of a sequential quota-based PR method
is how winner-voting ballots are reweighted. 

In the ssmpr.py implementation, we attempt to reweight in
proportion to the score, for those scores at or above the quota
threshold approval rating:

    QTASum = Sum(v*S[v] + (v+1)*S[v+1] + ... + maxscore * S[maxscore])
        for v = quota threshold approval rating

    Factor[r] = 1 - r * maxscore * Quota / QTASum

However, it is possible for Quota / Quota-Threshold-Approval to be
less than one while maxscore * Quota / QTASum is greater than one, so
in those cases, we successively reweight the top score to zero and
rescale the remaining portion of above-threshold rating ballots.
In the worst case, all but the quota-threshold-rating (score=v)
ballots will be reweighted to zero.
