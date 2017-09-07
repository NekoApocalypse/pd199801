import pickle
import re
import random

class Daily_Vocabulary(object):
    def __init__(self):
        self._dictionary_file = 'word_pure.dat'
        self._copura_file = '199801.txt'
        self.load_dictionary_pure()

    def load_dictionary_pure(self):
        """Load Dictionary file and convert copura to id sequence
        """
        with open(self._dictionary_file) as f:
            pickle.load(word_count, f)
            pickle.load(word_id, f)
        self.word2count = word_count
        self.word2id = word_id
        self.id2word = sorted(word_id.items(), key=lambda x:x[1])
        self.book = []
        with open(self._copura_file) as f:
            for line in f:
                line = re.sub(r'(\[)|(\]\w+)','',line)
                content = [
                    word2id[re.sub(r'\/\w+','',word)] for word in line.split()[1:]
                ]
                if len(content)>0:
                    self.book.append(content)

    def sample_output(self):
        tmp = random.choice(book)
        print(tmp)
        print([id2word[code] for code in tmp])

if __name__=='__main__':
    dict = Daily_Vocabulary()
    dict.sample_output()
