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
        with open(self._dictionary_file, 'rb') as f:
            word_count = pickle.load(f)
            word_id = pickle.load(f)
        self.word2count = word_count
        self.word2id = word_id
        self.id2word = sorted(word_id.items(), key=lambda x:x[1])
        self.id2word = [item[0] for item in self.id2word]
        self.book = []
        with open(self._copura_file) as f:
            for line in f:
                line = re.sub(r'(\[)|(\]\w+)','',line)
                content = [
                    self.word2id[re.sub(r'\/\w+','',word)] for word in line.split()[1:]
                ]
                if len(content)>0:
                    self.book.append(content)

    def sample_output(self):
        #key_list = [item[1] for item in self.word2id.items()]
        #print(max(key_list))
        tmp = random.choice(self.book)
        print(tmp)
        print([self.id2word[code] for code in tmp])


if __name__=='__main__':
    dict = Daily_Vocabulary()
    dict.sample_output()
