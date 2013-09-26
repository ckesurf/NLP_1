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

def simple_conll_corpus_iterator(corpus_file):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            # Extract information from line.
            # Each line has the format
            # word pos_tag phrase_tag ne_tag
            fields = line.split(" ")
            ne_tag = fields[-1]
            #phrase_tag = fields[-2] #Unused
            #pos_tag = fields[-3] #Unused
            word = " ".join(fields[:-1])
            yield word, ne_tag
        else: # Empty line
            yield (None, None)                        
        l = corpus_file.readline()

def sentence_iterator(corpus_iterator):
    """
    Return an iterator object that yields one sentence at a time.
    Sentences are represented as lists of (word, ne_tag) tuples.
    """
    current_sentence = [] #Buffer for the current sentence
    for l in corpus_iterator:        
            if l==(None, None):
                if current_sentence:  #Reached the end of a sentence
                    yield current_sentence
                    current_sentence = [] #Reset buffer
                else: # Got empty input stream
                    sys.stderr.write("WARNING: Got empty input file/stream.\n")
                    raise StopIteration
            else:
                current_sentence.append(l) #Add token to the buffer

    if current_sentence: # If the last line was blank, we're done
        yield current_sentence  #Otherwise when there is no more token
                                # in the stream return the last sentence.

def get_ngrams(sent_iterator, n):
    """
    Get a generator that returns n-grams over the entire corpus,
    respecting sentence boundaries and inserting boundary tokens.
    Sent_iterator is a generator object whose elements are lists
    of tokens.
    """
    for sent in sent_iterator:
         #Add boundary symbols to the sentence
         w_boundary = (n-1) * [(None, "*")]
         w_boundary.extend(sent)
         w_boundary.append((None, "STOP"))
         #Then extract n-grams
         ngrams = (tuple(w_boundary[i:i+n]) for i in xrange(len(w_boundary)-n+1))
         for n_gram in ngrams: #Return one n-gram at a time
            yield n_gram        


class Hmm(object):
    """
    Stores counts for n-grams and emissions. 
    """

    def __init__(self, n=3):
        assert n>=2, "Expecting n>=2."
        self.n = n
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.all_states = set()
        self.tag_frequency = defaultdict(int)

    def train(self, corpus_file):
        """
        Count n-gram frequencies and emission probabilities from a corpus file.
        """
        ngram_iterator = \
            get_ngrams(sentence_iterator(simple_conll_corpus_iterator(corpus_file)), self.n)

        for ngram in ngram_iterator:
            #Sanity check: n-gram we get from the corpus stream needs to have the right length
            assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (len(ngram, self.n))

            tagsonly = tuple([ne_tag for word, ne_tag in ngram]) #retrieve only the tags            
            for i in xrange(2, self.n+1): #Count NE-tag 2-grams..n-grams
                self.ngram_counts[i-1][tagsonly[-i:]] += 1
            
            if ngram[-1][0] is not None: # If this is not the last word in a sentence
                self.ngram_counts[0][tagsonly[-1:]] += 1 # count 1-gram
                self.emission_counts[ngram[-1]] += 1 # and emission frequencies

            # Need to count a single n-1-gram of sentence start symbols per sentence
            if ngram[-2][0] is None: # this is the first n-gram in a sentence
                self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

        # Create tag frequency and word count
        for word, ne_tag in self.emission_counts:
            self.tag_frequency[ne_tag] += self.emission_counts[(word, ne_tag)]


    def write_counts(self, output, printngrams=[1,2,3]):
        """
        Writes counts to the output file object.
        Format:

        """
        # First write counts for emissions
        for word, ne_tag in self.emission_counts:
            output.write("%i WORDTAG %s %s\n" % (self.emission_counts[(word, ne_tag)], ne_tag, word))


        # Then write counts for all ngrams
        for n in printngrams:            
            for ngram in self.ngram_counts[n-1]:
                ngramstr = " ".join(ngram)
                output.write("%i %i-GRAM %s\n" %(self.ngram_counts[n-1][ngram], n, ngramstr))

    def read_counts(self, corpusfile):

        self.n = 3
        self.emission_counts = defaultdict(int)
        self.ngram_counts = [defaultdict(int) for i in xrange(self.n)]
        self.all_states = set()

        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.all_states.add(ne_tag)
                self.tag_frequency[ne_tag] += self.emission_counts[(word, ne_tag)]
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM",""))
                ngram = tuple(parts[2:])
                self.ngram_counts[n-1][ngram] = count
        print self.tag_frequency

    def emission_params(self, x, y):
        test = self.emission_counts[(x, y)]/float(self.tag_frequency[y])
        emis_count = self.emission_counts[(x, y)]
        return self.emission_counts[(x, y)]/float(self.tag_frequency[y])

    def entity_tagger(self, x):
        # find the tag y with highest emission_params score
        # what are all the tags a word could have?
        best_tag = ""
        best_tag_prob = 0
        for tag in self.all_states:
            if self.emission_params(x, tag) > best_tag_prob:
                best_tag = tag
                best_tag_prob = self.emission_params(x, tag)
        return best_tag

                
def replace_rare(name):

    word_count = defaultdict(int)

    # create new file "ner_train_replaced.dat", copy filename contents to ntr file.
    filename = file(name, "r")
    l = filename.readline()
    while l:
        line = l.strip()
        if line: # Nonempty line
            # Calculate frequency of word
            fields = line.split(" ")
            word_count[fields[0]] += 1

        l = filename.readline()

    # print out word counts
    #print word_count

    new_training = "ner_train_replaced.dat"
    shutil.copyfile(name, new_training)


    # now replace all rare words
    new_training_tmp = "ner_train_replaced.dat.tmp"

    import os

    for word in word_count:
        if (word_count[word] < 5):
            tmp = open(new_training_tmp,"a") #open for append
            for l in open(new_training):
                line = l.strip()
                if line: # Nonempty line
                    fields = line.split(" ")
                    # word must match first term EXACTLY
                    if (fields[0] == word):
                        line = line.replace(word, "_RARE_")
                tmp.write(line + '\n')
            tmp.close()
            os.remove(new_training)
            os.rename(new_training_tmp, new_training)

def usage():
    print """
    python count_freqs.py [input_file] > [output_file]
        Read in a named entity tagged training input file and produce counts.
    """

if __name__ == "__main__":

    if len(sys.argv)!=2: # Expect exactly one argument: the training data file
        usage()
        sys.exit(2)

    try:
        input = file(sys.argv[1],"r")
    except IOError:
        sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
        sys.exit(1)
    #
    #
    #
    # Initialize a trigram counter
    # counter = Hmm(3)
    # # Collect counts
    # counter.train(input)
    # # Write the counts
    # counter.write_counts(sys.stdout)

    # Replace infrequent words (where Count(x) < 5) in input data file to _RARE_
    # replace_rare(sys.argv[1])

    # Print out entity tagger
    counter = Hmm(3)
    # # Read counts, training the Hmm
    counter.read_counts(input)
    # #
    best_tag = counter.entity_tagger("_RARE_")
    print best_tag