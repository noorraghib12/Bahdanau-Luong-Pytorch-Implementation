from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


SOS_token = 0
EOS_token = 1

class Lang:
    """
    Lang(self,name:str)
    Attributes:
        word2index {dict}: Dictionary for mapping word to vocab indexes
        index2word {dict}: Dictionary for mapping vocab index to words
        word2count {count}: Dictionary for mapping word to its frequency of appearance in dataset
    Methods:
        addSentence:
            args:
                sentence {str}
            Adds words from sentence to vocab 
        addWord
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

# MAX_LENGTH = 30

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p,reverse):
    if reverse is not True:
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH and \
            p[0].startswith(eng_prefixes)
    else:
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH and \
            p[1].startswith(eng_prefixes)

def filterPairs(pairs,reverse):
    return [pair for pair in pairs if filterPair(pair,reverse)]




def prepareData(lang1, lang2, reverse=False,test_size=None):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs,reverse)
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang,sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang,sentence):
    indexes=indexesFromSentence(lang,sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long,device=device).view(1,-1)

def tensorFromPair(pair):
    input_tensor=tensorFromSentence(input_lang,pair[0])
    output_tensor=tensorFromSentence(output_lang,pair[1])
    return(input_tensor,output_tensor)




def get_dataloader(lang1,lang2,batch_size,test_ratio=0.1):
    input_lang,output_lang,pairs=prepareData(lang1,lang2,True)
    N=len(pairs)
    input_ids=torch.zeros(size=(N,MAX_LENGTH),dtype=torch.long)
    output_ids=torch.zeros(size=(N,MAX_LENGTH),dtype=torch.long)

    for idx,(inp,trg) in enumerate(pairs):
        try:
            input_tensor,output_tensor=tensorFromPair((inp,trg))
        except KeyError:
            print(f"error at {idx}th pair")
            print(inp,trg)
            continue
        input_ids[idx,:input_tensor.shape[1]]=input_tensor
        output_ids[idx,:output_tensor.shape[1]]=output_tensor
    
    test_size=int(N*test_ratio)
        
    test_idx=np.random.randint(low=0,high=N,size=test_size)
    train_idx=np.setdiff1d(np.arange(N),test_idx)
    
    train_inp,train_outp=input_ids[train_idx],output_ids[train_idx]
    test_inp,test_outp=input_ids[test_idx],output_ids[test_idx]
    
    train_data=TensorDataset(torch.LongTensor(train_inp).to(device),
                             torch.LongTensor(train_outp).to(device)
                             )
    test_data=TensorDataset(torch.LongTensor(test_inp).to(device),
                             torch.LongTensor(test_outp).to(device)
                             )
    
    train_sampler=RandomSampler(train_data)
    train_dataloader=DataLoader(train_data,sampler=train_sampler,batch_size=batch_size)

    test_sampler=RandomSampler(test_data)
    test_dataloader=DataLoader(test_data,sampler=test_sampler,batch_size=batch_size)
    
    return input_lang,output_lang,train_dataloader,test_dataloader
