"""
author: Görkem Yılmaz
date: July 3, 2020
"""
import nlp_funcs
import pickle
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import sys


def main():
    # inputText = "Enter the input text here or give it as argument"
    inputText = str((sys.argv)[1]) # Input text can also be given as argument
    # Load the previously saved dataset
    myCorpus = pickle.load(open("theCorpus.sav", 'rb'))

    # tokend = pickle.load(open("word_tokenized_corpus.sav", 'rb'))

    # A function that prints the most similar sentences
    def output_sentences(most_similar):
        for label, index in [('1.', 0), ('2.', 1), ('3.', len(most_similar)//2), ('4.', len(most_similar) - 1)]:
            print(u'%s %s: %s\n' % (label, nlp_funcs.clearOnlyHTMLTags(most_similar[index][1]), nlp_funcs.clearOnlyHTMLTags(myCorpus[int(most_similar[index][0])])))
    # Load the previously trained and saved model
    model= Doc2Vec.load("trained.model")

    # To find the vector of a document which is not in training data
    test_data = word_tokenize(nlp_funcs.clearText(inputText))
    v1 = model.infer_vector(test_data)
    # print("V1_infer", v1)

    # To find most similar doc using tags
    similar_doc = model.docvecs.most_similar([v1])
    print(similar_doc)

    # Print the input text before the suggested sentences
    print("Input text is: " + inputText)
    # Print the most similar sentences
    output_sentences(similar_doc) 

    # To find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
    # print(model.docvecs['1'])

if __name__ == "__main__":
    main()