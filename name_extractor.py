'''
Extract Person Names (nr) from People's Daily Annotated Corpus
--------------------------------------------------------------
### IMPORTANT ASSUMPTION ###
    * A [/nr, /nr] sequence is a single person name. (first name + last name)
    x A [/nr, /n] sequence is a single person name (first name + title).
        ? This assumption is false. e.g. “彼得雕像”
        x This assumption is discarded. Title mention should be resolved by /
            anaphora resolution.
    * No name should follow another name.
'''

from collections import Counter
import pickle
import re
import io
import numpy as np
import matplotlib.pyplot as plt

def read_file(book_path, encoding='utf-8'):
    '''
    Read pos_word sequence from file.
    :param book_path: Path to text file in pd1998 format.
    :param encoding: Encoding of input file.
    :return: sentence_cat: [pos_word], list of pos_word.
    '''
    sentence_cat = []
    with open(book_path, encoding=encoding) as f:
        for line in f:
            lb = -1
            content = line.split()[1:]
            #print(content)
            for  i, raw_word in enumerate(content):
                if raw_word[0] == '[':
                    lb = i
                elif ']' in raw_word:
                    new_word = ' '.join(content[lb:i+1])
                    pos_word = (new_word.split(']')[0]+']', new_word.split(']')[1])
                    sentence_cat.append(pos_word)
                    lb = -1
                elif lb==-1:
                    pos_word = tuple(raw_word.split('/'))
                    sentence_cat.append(pos_word)
            # an empty pos_word is appended to sentence_cat to mark the end of a sentence
            sentence_cat.append(('','end'))
    return sentence_cat

def extract_nr(sentence_cat):
    '''
    :param sentence_cat: [pos_word], concatenated sentence consists of pos_word
    :return: nr_list:  [string], list of names
    '''
    nr_list = []
    skip = False
    for i, word in enumerate(sentence_cat[1:], start=1):
        prev_word = sentence_cat[i-1]
        if skip:
            skip = False
        elif prev_word[1] == 'nr':
            if word[1] == 'nr': #or word[1] == 'n':
                nr_entry = prev_word[0]+word[0]
                skip = True
            else:
                nr_entry = prev_word[0]
            nr_list.append(nr_entry)
    return nr_list

#def extract_nt(sentence_cat):

def write_file(var, output_path, encoding='utf-8'):
    with io.open(output_path, 'w+', encoding=encoding) as f:
        print(var, file=f)

def hist_from_dict(dict):
    counts = np.asarray(list(dict.values()))
    n, bins, patches =  plt.hist(counts, 50)
    plt.xlabel('Occurrences')
    plt.ylabel('Number of Names')
    plt.title('Distribution of Name Occurrences')
    plt.grid(True)
    plt.draw()

if __name__=='__main__':
    book_path = '199801.txt'
    #book_path = 'dummy.txt'
    test_output = 'test_output.txt'
    nr_output = 'names_extracted.txt'
    nt_output = 'organizations_extracted.txt'
    sentence_cat = read_file(book_path)
    nr_list = extract_nr(sentence_cat)
    nr_counted = Counter(nr_list)
    hist_from_dict(nr_counted)
    print('Total name occurrences: ', len(nr_list))
    print('Unique names: ',len(nr_counted))
    counts = list(nr_counted .values())
    write_file(nr_counted, test_output)
    plt.show()
