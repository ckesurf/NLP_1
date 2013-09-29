#! /usr/bin/python

__author__="Daniel Bauer <bauer@cs.columbia.edu>"
__date__ ="$Sep 12, 2011"

import sys
from collections import defaultdict
import math
import shutil

"""
Count n-gram frequencies in a CoNLL NER data file and write counts to
stdout. 
"""

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def replace_numeral(name):

    # create new file "ner_train.dat.replaced", copy filename contents to ntr file.
    try:
        filename = open(name,"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)

    word_count = defaultdict(int)
    l = filename.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            # Calculate frequency of word
            fields = line.split(" ")
            word_count[fields[0]] += 1

        l = filename.readline()
    filename.close()

    # now rewrite lines to new replacement file...

    filename = open(name, "r")
    new_training = name + ".replaced_numeral"
    o = open(new_training,"a")
    l = filename.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            fields = line.split(" ")
            # ... and replace rare words with _RARE_ and numerals with _NUMERAL_
            if (word_count[fields[0]] < 5):
                if is_number(fields[0]):
                    line = "_NUMERAL_ " + fields[1]
                else:
                    line = "_RARE_ " + fields[1]

        # Regardless, write line to new file
        o.write(line + '\n')
        l = filename.readline()
    filename.close()
    o.close()

def usage():
    print """
    python replace_numeral.py [input_file]
        Read in a named entity tagged training input file, replace rares to _RARE_ and numerals to _NUMERAL_,
        and rewrite to input_file.replaced_numeral
    """

if __name__ == "__main__":

    if len(sys.argv)!=2: # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    # Replace numerals to _NUMERAL_ and infrequent words (where Count(x) < 5) to _RARE_
    replace_numeral(sys.argv[1])


