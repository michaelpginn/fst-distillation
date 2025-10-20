# alignment
Module related to the Hulden alignment of a string-to-string dataset.

Mans's description:

Here's my original alignment algorithm description off the top of my head (the one in C, so you don't have to figure it out from the code):

(1) Start with a random alignment of all word-pairs
(2) Collect counts from these alignments (e.g. x:y seen 5 times, x:0 seen 4 times, etc.)
(3) Iterate over corpus as follows:
    (a) pick a pair
	(b) fill in costs in MED trellis (using the alignment counts, e.g. p(x:y) = count(x:y) / totalcount(allpairs)) [+ some smoothing]. Uses negative logprobs to convert probs to "cost"
    (c) start from the end (top right in my implementation) of the trellis and sample a path to the beginning. Each sampling step can move left, down, or diagonally. The steps are chosen by a weighted coin toss based on the probabilities of deleting, inserting, or matching the characters in question.
	(d) the pair may now have a new alignment; adjust counts based on this.

(b) is this line in the code:

cost = (double)( g_current_count[in][out] + g_prior ) / (double)( g_paircount + g_distinct_pairs * g_prior );

Later converted to negative logprobs.

The potential theoretical weakness in this algorithm is step (c) where you sample locally instead of globally from all the possible alignments of the pair. One could use forward-backward to add fractional counts instead (in which case the algorithm becomes like soft EM), but that sacrifices speed.

Last step (4): once counts/probs converge in (3), align corpus using the best path for each pair, output alignments.
