### Corpus structure is defined as following:
The basic elements are **word** and **postag**.

* (word, postag) :  A tuple like this is a **posword**.
* {pos_word}。 : A sequence of pos_words ending with a comma is a period is a **sentence**.
* {sentence}\n : A sequence of sentences ending with a linebreak is an **article**. 
* {article} : A sequence of articles contained in a file is a **book**. 
