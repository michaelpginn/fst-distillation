This experiment closely follows the standard activation clustering paradigm of Giles et al (1991) and the L* approach of Weiss et al. (2018). However, these methods are designed for extracting acceptors, so we need to modify the approach to work for transducers:

1. Use a prebuilt aligner (Hulden alignment) to predict (input symbol, output symbol) alignments. 
2. Train an RNN binary acceptor on tuple sequences.
3. Perform clustering and FSA extraction.
4. Convert the FSA to an FST by converting (input symbol, output symbol) tuples into transductions.
