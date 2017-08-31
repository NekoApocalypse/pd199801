from collections import Counter
import re


vocabulary = []
composite_vocab = []
with open('199801.txt') as f:
    for line in f:
        complex_word = re.findall(r'\[[^\]]+\]\w+',line)
        composite_vocab.extend(complex_word)
        line = re.sub(r'(\[)|(\]\w+)','',line)
        content = line.split()[1:]
        if(len(content)>0):
            vocabulary.extend(content)

vocabulary_pos = [tuple(word.split('/')) for word in vocabulary]
vocabulary_counted = Counter(vocabulary_pos)

with open('vocabulary.txt','w+') as f:
    for [word,pos],count in vocabulary_counted.most_common():
        f.write(word+'/'+pos+'/'+str(count)+'\n')
    for word in composite_vocab:
        f.write(word+'\n')
