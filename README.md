# Approval Sorted Margins
## A Condorcet-completion method with explicit approval cutoff

Approval Sorted Margins was introduced by Forest Simmons as a symmetric
modification to Definitive Majority Choice (AKA Ranked Approval Voting).

https://electowiki.org/wiki/Approval_Sorted_Margins

Recently I've been putting more effort into a variant that applies Sorted
Margins on a seed of Total Scores instead of Approval with cutoff. I think
that with sufficient range, for example zero to 10, this approach gives
virtually the same benefit as Approval Sorted Margins, with the advantage that
it can be used in a form of Quota-based Score Reweighted Voting.

Therefore, I've developed the code in

    ScoreSM.py

which will run Score Sorted Margins (ScoreSM) as a Droop Proportional
multiwinner election.  The first candidate seated is always the single-winner
ScoreSM winner.  If a seat winner does not have at least one quota of
approval, ballots approving of the seat winner are reweighted to remove one
quota, and a runner-up is found on the remaining reweighted ballots.

Example usage (add -v for verbose level 1, -vv for verbose level 2, etc.

	ScoreSM.py -vv -i examples/rlg_approval_cutoff_3.csv

Multiwinner, 9 seats:

	ScoreSM.py -m 9 -vv -i examples/june2011.csv

RCV election in Minnesota, 2013:

	ScoreSM.py -t rcv -vv -i examples/actual-mayoral-election.csv
