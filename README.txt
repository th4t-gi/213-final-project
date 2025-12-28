
+----------------------------------------------------------------+
|   A Parallel Algorithm to Compute the Canonical Partition      |
|           Function for p-adic log-Coloumb Gases                |
+----------------------------------------------------------------+

This algorithm computes the canonical partition function for log-Coloumb gas model for a given set of particles (and their corresponding charges)

USAGE:

>  build/main [OPTIONS]

Options:
    --help Show this help message and exit
    -q CHARGES Comma-separated (and no spaces) list of charges for each particle (float),
        e.g. -q 1,-1,3.1415,100
    -p PRIMES Comma-separated list of primes (integers),
        e.g. -p 2,3,11,
    -b BETA_STEP Beta step (positive float),
        e.g. -b 0.1

Defaults:
    charges: -1,-1,1,1
    primes: 2,3,5
    beta_step: 0.1


EXAMPLE:

> build/main -q 1,-1,1,-1,1,-1,1,-1 -p 2,3,5 -b 0.05
