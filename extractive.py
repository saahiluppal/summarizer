# Loading necessary libraries
import numpy as np
import networkx as nx

import spacy
nlp = spacy.load('en_core_web_lg')

import warnings
warnings.filterwarnings('ignore')


def textrank(filename):
    with open(filename) as f:
        output = f.read()

    output = output.replace('\n',' ')

    doc = nlp(output)

    sentences = [sent for sent in doc.sents]
    print("Total sentences found",len(sentences))

    print('Analyzing')

    clean_sentences = list()
    for sent in sentences:
        clean_sentences.append(" ".join([token.text for token in sent if not nlp.vocab[token.text].is_stop]))

    clean_sentences = [s.lower() for s in clean_sentences]

    sim_mat = np.zeros([len(sentences),len(sentences)])

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i!=j:
                sim_mat[i][j] = nlp(clean_sentences[i]).similarity(nlp(clean_sentences[j]))
                print(i,j,end='\r')

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_senteces = sorted(((scores[i],s) for i,s in enumerate(sentences)),reverse=True)

    print('Summary::')
    for i in range(6):
        print(ranked_senteces[i][1],end=' ')
    print()
