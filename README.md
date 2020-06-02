# Approval Sorted Margins
## A Condorcet-completion method with explicit approval cutoff

Approval Sorted Margins was introduced by Forest Simmons as a symmetric
modification to Definitive Majority Choice (AKA Ranked Approval Voting).

https://electowiki.org/wiki/Approval_Sorted_Margins

Code for this, compared with other approval cutoff schemes, may be found
in

    asm.py

A variant that applies Sorted Margins on a seed of Total Scores, instead
of Approval with cutoff, may be used in a form of Quota-based threshold
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
