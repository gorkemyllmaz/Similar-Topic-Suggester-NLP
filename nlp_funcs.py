"""
author: Görkem Yılmaz
date: June 29, 2020
"""

import re
import os
import pandas as pd
import csv
import pickle
from keras.preprocessing.text import Tokenizer
from gensim.models.fasttext import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
from sklearn.decomposition import PCA

import nltk
import nltk.corpus
from nltk import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize


# This function clears the texts and removes stopwords
def clearText(text, remove_stopwords=True, stem_words=False):
    # Remove Questions: and Details: from the word tokenized corpus
    text = re.sub("Questions:", "", text)
    text = re.sub("Details:", "", text)
    text = re.sub(re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});"), "", text)
    
    # Remove punctuation from text
    text.translate(str.maketrans('', '', punctuation))

    # Remove stopwords from text
    if remove_stopwords:
        text = re.sub("but", "", text)
        text = re.sub("there", "", text)
        text = re.sub("about", "", text)
        text = re.sub("ourselves", "", text)
        text = re.sub("hers", "", text)
        text = re.sub("between", "", text)
        text = re.sub("yourself", "", text)
        text = re.sub("again", "", text)
        text = re.sub("once", "", text)
        text = re.sub("out", "", text)
        text = re.sub("during", "", text)
        text = re.sub("very", "", text)
        text = re.sub("having", "", text)
        text = re.sub("with", "", text)
        text = re.sub("they", "", text)
        text = re.sub("own", "", text)
        text = re.sub("an", "", text)
        text = re.sub("be", "", text)
        text = re.sub("some", "", text)
        text = re.sub("for", "", text)
        text = re.sub("do", "", text)
        text = re.sub("its", "", text)
        text = re.sub("yours", "", text)
        text = re.sub("such", "", text)
        text = re.sub("into", "", text)
        text = re.sub("of", "", text)
        text = re.sub("most", "", text)
        text = re.sub("itself", "", text)
        text = re.sub("other", "", text)
        text = re.sub("off", "", text)
        text = re.sub("is", "", text)
        text = re.sub("s", "", text)
        text = re.sub("am", "", text)
        text = re.sub("or", "", text)
        text = re.sub("who", "", text)
        text = re.sub("as", "", text)
        text = re.sub("from", "", text)
        text = re.sub("him", "", text)
        text = re.sub("each", "", text)
        text = re.sub("the", "", text)
        text = re.sub("themselves", "", text)
        text = re.sub("until", "", text)
        text = re.sub("below", "", text)
        text = re.sub("are", "", text)
        text = re.sub("we", "", text)
        text = re.sub("these", "", text)

        text = re.sub("your", "", text)
        text = re.sub("his", "", text)
        text = re.sub("through", "", text)
        text = re.sub("don", "", text)
        text = re.sub("nor", "", text)
        text = re.sub("me", "", text)
        text = re.sub("were", "", text)
        text = re.sub("her", "", text)
        text = re.sub("more", "", text)
        text = re.sub("himself", "", text)
        text = re.sub("this", "", text)
        text = re.sub("down", "", text)
        text = re.sub("should", "", text)
        text = re.sub("our", "", text)
        text = re.sub("their", "", text)
        text = re.sub("while", "", text)
        text = re.sub("above", "", text)
        text = re.sub("both", "", text)
        text = re.sub("up", "", text)
        text = re.sub("to", "", text)
        text = re.sub("ours", "", text)
        text = re.sub("had", "", text)
        text = re.sub("she", "", text)
        text = re.sub("all", "", text)
        text = re.sub("no", "", text)

        text = re.sub("when", "", text)
        text = re.sub("at", "", text)
        text = re.sub("any", "", text)
        text = re.sub("before", "", text)
        text = re.sub("them", "", text)
        text = re.sub("same", "", text)
        text = re.sub("and", "", text)
        text = re.sub("been", "", text)
        text = re.sub("have", "", text)
        text = re.sub("in", "", text)
        text = re.sub("will", "", text)
        text = re.sub("on", "", text)
        text = re.sub("does", "", text)
        text = re.sub("yourselves", "", text)
        text = re.sub("then", "", text)
        text = re.sub("that", "", text)
        text = re.sub("because", "", text)
        text = re.sub("what", "", text)
        text = re.sub("over", "", text)
        text = re.sub("why", "", text)
        text = re.sub("so", "", text)
        text = re.sub("can", "", text)
        text = re.sub("did", "", text)
        text = re.sub("not", "", text)
        text = re.sub("now", "", text)
        text = re.sub("under", "", text)
        text = re.sub("he", "", text)
        text = re.sub("you", "", text)
        text = re.sub("herself", "", text)
        text = re.sub("has", "", text)
        text = re.sub("just", "", text)
        text = re.sub("where", "", text)
        text = re.sub("too", "", text)
        text = re.sub("only", "", text)
        text = re.sub("myself", "", text)
        text = re.sub("which", "", text)
        text = re.sub("those", "", text)
        text = re.sub("i", "", text)
        text = re.sub("after", "", text)

        text = re.sub("few", "", text)
        text = re.sub("whom", "", text)
        text = re.sub("being", "", text)
        text = re.sub("if", "", text)
        text = re.sub("theirs", "", text)
        text = re.sub("my", "", text)
        text = re.sub("against", "", text)
        text = re.sub("a", "", text)
        text = re.sub("by", "", text)
        text = re.sub("doing", "", text)
        text = re.sub("it", "", text)
        text = re.sub("how", "", text)
        text = re.sub("further", "", text)
        text = re.sub("was", "", text)
        text = re.sub("here", "", text)
        text = re.sub("than", "", text)


    # # Optionally, shorten words to their stems
    # if stem_words:
    #     text = text.split()
    #     stemmer = SnowballStemmer('english')
    #     stemmed_words = [stemmer.stem(word) for word in text]
    #     text = " ".join(stemmed_words)

    return text

# This function creates a corpus by reading a dataset
def createCorpus():

    # Read the dataset and add it to following array to create corpus
    theCorpus = []
    # Read entire dataset
    with open('/content/drive/My Drive/Colab Notebooks/JotformProje/SupportForumDataSetGorkem.csv', 'r') as file:
        reader = csv.reader(file)
        counter = 0
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            if row[1] != "" and row[2] != "":
                theCorpus.append("Question: " + row[1] + "\n" + "Details: " + row[2])
            
    print("READING THE DATASET IS DONE")
    # Save original corpus
    pickle.dump(theCorpus, open("theCorpus.sav", 'wb'))

    print("CREATING CORPUS")
    # Creating final corpus from the cleared corpus
    
    final_corpus = []
    cnt = 0
    for sentence in theCorpus:
        if sentence.strip() != '':
            final_corpus.append(clearText(sentence.lower()))
        cnt += 1
        print(cnt)
    word_punctuation_tokenizer = nltk.WordPunctTokenizer()
    # Tokenizing the corpus
    word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in final_corpus]
    print("CORPUS CREATED!")

    # Saving word tokenized corpus
    pickle.dump(word_tokenized_corpus, open("word_tokenized_corpus.sav", 'wb'))

    
# This function generates a graph that shows the closeness of the words in a dataset by taking some keywords
def visualizeWordDistance():
    
    # Load the previously saved corpus corpus (If no file found, createCorpus function must be ran)
    word_tokenized_corpus = pickle.load(open("word_tokenized_corpus.sav", 'rb'))

    # Creating fasttext model
    ft_model = FastText(word_tokenized_corpus,
                    size=60,
                    window=40,
                    min_count=5,
                    sample=1e-2,
                    sg=1,
                    iter=100)

    print("FASTTEXT MODEL CREATED")
    
    # print(ft_model.wv['form'])

    print("CHECKING SIMILAR WORDS")
    # Use some keywords to find closeness of words in dataset
    semantically_similar_words = {words: [item[0] for item in ft_model.wv.most_similar([words], topn=5)]
                for words in ['internet', 'jotform', 'button', 'submit', 'issue']}

    # for k,v in semantically_similar_words.items():
    #     print(k+":"+str(v))

    # print(ft_model.wv.similarity(w1='button', w2='submission'))

    print("STARTING DATA VISUALIZATION")
    # Collect all similar words
    all_similar_words = sum([[k] + v for k, v in semantically_similar_words.items()], [])

    # print(all_similar_words)
    # print(type(all_similar_words))
    # print(len(all_similar_words))

    # Vectorize the words
    word_vectors = ft_model.wv[all_similar_words]
    pca = PCA(n_components=2)
    p_comps = pca.fit_transform(word_vectors)
    word_names = all_similar_words

    # Draw the pilot
    plt.figure(figsize=(18, 10))
    plt.scatter(p_comps[:, 0], p_comps[:, 1], c='red')

    for word_names, x, y in zip(word_names, p_comps[:, 0], p_comps[:, 1]):
        plt.annotate(word_names, xy=(x+0.06, y+0.03), xytext=(0, 0), textcoords='offset points')

    plt.title("Word Closeness Graph")
    plt.show()

# This function only clears HTML tags e.g <p> or <\br>
def clearOnlyHTMLTags(text):
    # Empty question
    if type(text) != str or text=='':
        return ''

    # Clean the text
    text = re.sub(re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});"), "", text)
    # Return cleared text
    return text


# This function trains a model to find similar sentences to the input sentence, then suggests related sentences from dataset.
def suggestSimilar(inputText):

    # Load the previously saved dataset
    myCorpus = pickle.load(open("theCorpus.sav", 'rb'))
    # tokened = pickle.load(open("word_tokenized_corpus.sav", 'rb'))

    # A function that prints the most similar sentences
    def output_sentences(most_similar):
        for label, index in [('1.', 0), ('2.', 1), ('3.', len(most_similar)//2), ('4.', len(most_similar) - 1)]:
            print(u'%s %s: %s\n' % (label, clearOnlyHTMLTags(most_similar[index][1]), clearOnlyHTMLTags(myCorpus[int(most_similar[index][0])])))

    
    ## Trainig Starts

    # Get the tagged data by using TaggedDocument function
    tagged_data = [TaggedDocument(words=word_tokenize(clearText(_d.lower())), tags=[str(i)]) for i, _d in enumerate(tokend)]
    max_epochs = 200
    vec_size = 20
    alpha = 0.025
    # Create a Doc2Vec model to train
    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =1)
    # Build vocabulary from tagged data
    model.build_vocab(tagged_data)

    # Train the model for specified epoch number
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch+1))
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        # Decrease the learning rate
        model.alpha -= 0.0002
        # Fix the learning rate, no decay
        model.min_alpha = model.alpha

    # Save the trained model
    model.save("trained.model")
    print("Model Saved")
    # TRAIN Ends

    # Load the previously trained and saved model
    model= Doc2Vec.load("trained.model")
    
    # To find the vector of a document which is not in training data
    test_data = word_tokenize(clearText(inputText.lower()))
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