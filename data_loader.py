import pickle
import re
import random

class Daily_Vocabulary(object):
    def __init__(self):
        self._dictionary_file = 'word_pure.dat'
        self._copura_file = '199801.txt'
        self._window_size = 5
        self._batch_size  = 40
        self.load_dictionary_pure()
        self.label = 0
        self.end_of_epoch = False

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
        self.id2count = [word_count[word] for word in self.id2word]
        self.book = []
        with open(self._copura_file) as f:
            for line in f:
                line = re.sub(r'(\[)|(\]\w+)','',line)
                content = [
                    self.word2id[re.sub(r'\/\w+','',word)] for word in line.split()[1:]
                ]
                if len(content)>0:
                    self.book.extend(content)
        self._book_size = len(self.book)

    def restart_epoch(self):
        self.end_of_epoch = False
        self.label = 0

    def generate_batch(self):
        count = 0
        x = []
        y = []
        while(count < self._batch_size):
            center = self.label
            self.label = self.label + 1
            for inc in range(1, self._window_size):
                if(center + inc < self._book_size):
                    x.append(self.book[center])
                    y.append(self.book[center+inc])
                    count = count + 1
            if self.label==self._book_size:
                self.end_of_epoch = True
                break
        return x,y


if __name__=='__main__':
    dict = Daily_Vocabulary()