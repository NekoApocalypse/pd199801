from collections import Counter
import pickle
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
#Dictionary: {(word,pos_tag):count}
vocabulary_counted = Counter(vocabulary_pos)
#Dictionary: {(word,pos_tag):id}
vocabulary_id = dict(
    zip(sorted(vocabulary_pos, key=lambda x:vocabulary_counted[x]),
        range(1, len(vocabulary_pos)+1)
    )
)

vocabulary_pure = [word[0] for word in vocabulary_pos]
pure_counted = Counter(vocabulary_pure)
pure_id = dict(
    zip(sorted(vocabulary_pure, key=lambda x:pure_counted[x]),
        range(1, len(vocabulary_pure)+1)
    )
)

with open('word_pos.dat', 'wb') as f:
    pickle.dump(vocabulary_counted, f)
    pickle.dump(vocabulary_id, f)

with open('word_pure.dat', 'wb') as f:
    pickle.dump(pure_counted, f)
    pickle.dump(pure_id, f)

with open('vocabulary.txt','w+') as f:
    for [word,pos],count in vocabulary_counted.most_common():
        f.write(word+'/'+pos+'/'+str(count)+'\n')
    for word in composite_vocab:
        f.write(word+'\n')
