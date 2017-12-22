# pybaseline #

This is an algorithm for extracting the baseline from a signal containing quantal data. The algorithm was originally described in "Automated Maximum Likelihood Separation of Signal from Baseline in Noisy Quantal Data" by William J. Bruno. The algorithm was published along with C code by the authors. This is a manual translation of the code into python. Numba is used when necessary to improve execution speed. Sections of code are rewritten for a more pythonic style.
