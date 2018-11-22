from ctypes import cdll, POINTER, c_int, c_double
from os.path import abspath
import numpy as np

# set up for future calls to c
_fit = cdll.LoadLibrary(abspath("lda_gibbs.so")).fit
_fit.argtypes = (c_int, *([POINTER(c_int)]*5), *([c_int]*4), *([c_double]*2))

# function to help handle ndarrays
def _np2c(x):
    return x.ctypes.data_as(POINTER(c_int))

class LDA:
    
    def __init__(self, n_topics, n_iters=1000, alpha=None, beta=None):
        
        # good defaults
        if (alpha is None) or (alpha <= 0):
            alpha = 50/n_topics
        if (beta is None) or (beta <= 0):
            beta = 0.01
            
        # params
        self.alpha = alpha
        self.beta = beta
        self.n_topics = n_topics
        self.n_iters = n_iters
        
    def __str__(self):
        return f"LDA with {self.n_topics} topics"
    
    def __repr__(self):
        return f"LDA with {self.n_topics} topics"
        
    def fit(self, doc_term):
        
        # parse doc-term matrix
        n_docs, n_words = doc_term.shape
        n_tokens = doc_term.sum()
        docs, words = doc_term.nonzero()
        counts = np.array(doc_term[docs,words]).ravel()
        
        # get by token list of elements
        docs = np.repeat(docs, counts).astype(np.int32)
        words = np.repeat(words, counts).astype(np.int32)
        topics = np.random.randint(0,self.n_topics,size=words.size).astype(np.int32)
        
        # create counters
        C_wz = np.zeros((n_words, self.n_topics),dtype=np.int32)
        C_dz = np.zeros((n_docs, self.n_topics), dtype=np.int32)
        np.add.at(C_wz, (words,topics), 1)
        np.add.at(C_dz, (docs, topics), 1)
        
        # train the model
        _fit(self.n_iters,
             *[_np2c(x) for x in [C_wz,C_dz,words,docs,topics]], 
             n_tokens, n_words, n_docs, 
             self.n_topics, self.alpha, self.beta)
        
        # save user-topic distribution
        self.theta = (C_dz+self.alpha)/(np.sum(C_dz,axis=1)+self.n_topics*self.alpha)[:,None]
        
        # save topic-word distibution
        self.phi = ((C_wz+self.beta)/(np.sum(C_wz,axis=0)+n_words*self.beta)[None,:]).T
        
        return self