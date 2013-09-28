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
        self.tag_dict = defaultdict(list)
        self.word_count = defaultdict(int)
        self.pi_dict = dict([((0, '*', '*'), 1)])

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
        self.all_words = set()

        for line in corpusfile:
            parts = line.strip().split(" ")
            count = float(parts[0])
            if parts[1] == "WORDTAG":
                ne_tag = parts[2]
                word = parts[3]
                self.emission_counts[(word, ne_tag)] = count
                self.all_states.add(ne_tag)
                self.tag_frequency[ne_tag] += self.emission_counts[(word, ne_tag)]
                self.all_words.add(word)
                self.tag_dict[word].append(ne_tag)
                self.word_count[word] += 1
            elif parts[1].endswith("GRAM"):
                n = int(parts[1].replace("-GRAM",""))
                ngram = tuple(parts[2:])
                self.ngram_counts[n-1][ngram] = count

        self.tag_dict["*"] = "*"

    def emission_params(self, x, y):
        # if x == "the" and y == "D":
        #     return 0.8
        # elif x == "dog" and y == "D":
        #     return 0.2
        # elif x == "the" and y == "N":
        #     return 0.2
        # elif x == "dog" and y == "N":
        #     return 0.8
        # elif x == "barks" and y == "V":
        #     return 1

        if x == "*" and y == "*":
            return 1
        else:
            return self.emission_counts[(x, y)]/float(self.tag_frequency[y])

    def entity_tagger(self, x):
        # find the tag with highest emission_params score
        best_tag = ""
        best_tag_prob = 0
        for tag in self.all_states:
            if self.emission_params(x, tag) > best_tag_prob:
                best_tag = tag
                best_tag_prob = self.emission_params(x, tag)
        return best_tag

    def trigram_prob(self, y1, y2, y3):
        # if y1 == "*" and y2 == "*" and y3 == "D":
        #     return 1
        # elif y1 == "*" and y2 == "D" and y3 == "N":
        #     return 1
        # elif y1 == "D" and y2 == "N" and y3 == "V":
        #     return 1
        # elif y1 == "N" and y2 == "V" and y3 == "STOP":
        #     return 1
        count_y1_y2_y3 = self.ngram_counts[2][(y1, y2, y3)]
        count_y1_y2 = float(self.ngram_counts[1][(y1, y2)])
        if count_y1_y2 == 0 or count_y1_y2_y3 == 0:
            return 0
        return count_y1_y2_y3/count_y1_y2

    def pi(self, index, u, v, sentence):
        if index == 0 and u == '*' and v == '*':
            return 1
        elif (index, u, v) in self.pi_dict:        # if we've already calculated this, return the corresponding prob
            return self.pi_dict[(index, u, v)]
        else:
            max_prob = prob = 0
            for t in self.tag_dict[sentence[index-2]]:
                # if trigram_prob or emission params is 0, skip
                tri = self.trigram_prob(t, u, v)
                emission = self.emission_params(sentence[index], v)
                if tri == 0 or emission == 0:       # We KNOW the resulting prob will always be nonzero
                    continue
                else:
                    prob = self.pi(index-1, t, u, sentence)*tri*emission
                    if prob > max_prob:
                        max_prob = prob
            self.pi_dict[(index, u, v)] = max_prob
            return max_prob

    def bp(self, k, u, v, sentence):
        max_prob = prob = 0
        best_tag = ""
        for w in self.tag_dict[sentence[k-2]]:
            prob = self.pi(k-1, w, u, sentence)*self.trigram_prob(w, u, v)*self.emission_params(sentence[k], v)
            if prob > max_prob:
                max_prob = prob
                best_tag = w
        return best_tag

    def viterbi(self, sentence):
        length = sentence.__len__() - 2     # -2 because we added two * strings
        # want k to go from 1 to n; range is exclusive, so following translates to [1, n]

        for k in range(1, length+1):
            for u in self.tag_dict[sentence[k-1]]:
                for v in self.tag_dict[sentence[k]]:
                    self.pi(k, u, v, sentence)
                    prob = self.pi_dict[(k, u, v)]

        max_prob = prob = 0
        best_u_tag = best_v_tag = ""
        for u in self.tag_dict[sentence[k-1]]:
            for v in self.tag_dict[sentence[k]]:
                p = self.pi_dict[(length, u, v)]
                t = self.trigram_prob(u, v, "STOP")
                prob = self.pi_dict[(length, u, v)]*self.trigram_prob(u, v, "STOP")
                if prob > max_prob:
                    max_prob = prob
                    best_u_tag = u
                    best_v_tag = v

        tags = {length-1: best_u_tag, length: best_v_tag, 0: "*"}
        for k in range(length-2, 0, -1):
            tags[k] = self.bp(k+2, tags[k+1], tags[k+2], sentence)



        return tags

    def viterbi_file(self, filename):
        orig_sentence = list("*")
        sentence = list("*")
        test_file = file(filename, "r")
        l = test_file.readline()
        while l:
            word = l.strip()
            if word: # Nonempty line
                orig_sentence.append(word)
                # if rare, replace with _RARE_
                if self.word_count[word] < 5:
                    sentence.append("_RARE_")
                else:
                    sentence.append(word)

            else: # Empty line
                orig_sentence.append("*")
                sentence.append("*")
                # Viterbi time! Write probabilities and tags to stdout, then clear data
                tags = self.viterbi(sentence)
                for k in range(1, sentence.__len__()-1):
                    print("%s %s %f" % (orig_sentence[k], tags[k], math.log(self.pi_dict[k, tags[k-1], tags[k]])))
                print
                # garbage collection
                del tags
                self.pi_dict.clear()
                orig_sentence = list("*")
                sentence = list("*")
            l = test_file.readline()




def usage():
    print """
    python count_freqs.py [input_file] > [output_file]
        Read in a named entity tagged training input file, train Hmm, test on dev data, then print
        word, tag, and log probabilities.
    """

if __name__ == "__main__":

    if len(sys.argv)!=3: # Expect exactly two arguments: the training counts file and development (test) file
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
    counter = Hmm(3)
    # Read counts, training the Hmm
    counter.read_counts(input)
    # #


    res = counter.viterbi_file(sys.argv[2])
    print res

