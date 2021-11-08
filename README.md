# data-cleaning-example
Date: 17/09/2021

Environment: Python 3.7.9 and Anaconda 2020.11

Libraries used:
 * pandas
 * numpy
 * matplotlib
 * re
 * datetime
 * ast
 * networkx
 * sklearn
 * nltk
 * sympy
 * collections


This code deals with the cleaning of synthetic data designed to simulate real data produced by food delivery companies (such as uber eats). The data is extracted from 3 separate files, these contain dirty data (e.g. corrupted data), missing data, and outlier data. the processing of these files is executed separately and is split into 5 main sections:

 * Part 0 - importing data and libraries
 * Part 1 - imputing missing data 
 * Part 2 - filtering outlier data
 * Part 3 - cleaning dirty data
 * Part 4 - exporting data
 
 Parts 1 to 3 deal with the processing of the data. The processes are outlined in more detail within each section.
