#@title Load the Universal Sentence Encoder's TF Hub module
from __future__ import print_function
from ftplib import all_errors
from lib2to3.pgen2 import tokenize
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import pathlib
import seaborn as sns
import nltk
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Create WordNetLemmatizer object
wnl = WordNetLemmatizer()
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
model = hub.load(module_url)
print ("module %s loaded" % module_url)

class Sentences:
      def __init__(self, content, id, sentence_embedding, para_order, content_original):
            self.id = id
            self.content = content
            self.sentence_embedding = sentence_embedding
            self.content_original = content_original

            # the location of the sentence in the article
            self.para_order = para_order

            #sentence scores subpart
            self.title_method_score = None
            self.location_method_score = None
            self.sentence_length_score = None
            self.numerical_token_score = None
            self.proper_noun_score = None
            self.similarity_centroid = None
            self.keyword_frequency = None
            self.reduce_frequency_score = None

            #sentence score/weight
            self.sentence_score = None

      # print instance for debugging purpose
      def __repr__(self):
            # !!! for now just show the first three sentence embedding for the purpose of reduce the computing power used
            return "id: {} content: {} sentence_embedding: {}".format(self.id, self.content, self.sentence_embedding[:3]) + "\ntitle: {}, location:{}, sentence length:{}, numerical_token:{}, proper_noun:{}, similarity_centroid:{}, keyword_frequency:{}, kmeans:{}, sentence score:{}".format(self.title_method_score, self.title_method_score, self.sentence_length_score, self.numerical_token_score, self.proper_noun_score, self.similarity_centroid, self.keyword_frequency, self.reduce_frequency_score ,self.sentence_score)


class News:
  def __init__(self, category = None, title = None, sentences = None, para_order = None, path = None):
    self.path = path
    self.category = category
    self.sentences = sentences
    self.para_order = para_order
    # sentences, index corresponding to sentence id. eg: title has index 0, id 0
    self.content = []
    # get the sentence embedding for every sentence in the article
    sentences_embedding = News.populate_sentence_embedding(title, sentences)
    for i, embedding in enumerate(sentences_embedding):
        if (i == 0):
          self.title = Sentences(removeStopwordsTitle(title), i, embedding, 1, title)
          self.content.append(self.title)
        else:
          self.content.append(Sentences(removeStopwordsTitle(sentences[i-1]), i, embedding, para_order[i], sentences[i-1]))
    # caluclate the sub-part and sum of sentence score
    self.title_method()
    self.location_method()
    self.get_sentence_length_score()
    self.get_numerical_token_score()
    self.get_proper_noun_score()
    # self.reduce_frequency_helper(sentences_embedding)
    self.populate_sentence_score()

  def get_numerical_token_score(self):
    for sentence in self.content:
        words_in_sentence = sentence.content.split(" ")
        num_numerical_token = len(list(filter(lambda x:x.isdigit(), words_in_sentence)))
        sentence.numerical_token_score = num_numerical_token/len(words_in_sentence)


  def reduce_frequency_helper(self, sentences_embedding):
    # k to be decided
    k = int(0.4 * (len(self.content)-1))
    # get all the sentence embedding except for the title in the news article
    X = sentences_embedding[1:]
    labels = News.get_reduce_frequency_score(k, X)
    freq_count = Counter(labels)
    for i in range(len(self.content)):
      if i == 0:
        self.content[i].reduce_frequency_score = 1
      else:
        self.content[i].reduce_frequency_score = freq_count[labels[i-1]]/len(labels)

  
  def get_reduce_frequency_score(num_cluster, X):
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X)
    return kmeans.labels_

  def get_similarity_centroid_score(self):
    pass

  def get_keyword_frequency_score(self):
    pass

# Andy is a good Chinese Wow Boy.

  def get_proper_noun_score(self):
    for sentence in self.content:
      tagged_sentence = nltk.pos_tag(nltk.word_tokenize(sentence.content))
      propernouns = []
      prev_tag_NNP = False
      for word,pos_tag in tagged_sentence:
        if pos_tag == 'NNP' and not prev_tag_NNP:
          propernouns.append(word)
          prev_tag_NNP = True
        elif pos_tag == 'NNP' and prev_tag_NNP:
          propernouns[-1] += word
        else:
          prev_tag_NNP = False
      sentence.proper_noun_score = len(propernouns)/len(sentence.content.split(" "))

  def get_sentence_length_score(self):
    # find the longest sentence
    max_length = max([len(sentence.content.split(" ")) for sentence in self.content])
    for i in range(len(self.content)):
        if (i == 0):
          # title is set to 1
          self.content[i].sentence_length_score = 1
        else:
          self.content[i].sentence_length_score = len(self.content[i].content.split(" "))/max_length
    
     
  def populate_sentence_score(self):
    for sentence in self.content:
        try:
          sentence.sentence_score = sentence.location_method_score * 0.3 + sentence.title_method_score * 0.5 + sentence.sentence_length_score * 0.05 + sentence.numerical_token_score * 0.05 + sentence.proper_noun_score * 0.1
        except TypeError:
          print(sentence)
        
  def title_method(self):
    # calcluate the similarity of title to itself for later normalization
    title_self_similarity = np.inner(self.title.sentence_embedding, self.title.sentence_embedding)
    for sentence in self.content:
      # calculate the similarity of title and each sentence
      sentence.inner_product_title = np.inner(self.title.sentence_embedding, sentence.sentence_embedding)
      # normalize to the range of 0-1
      sentence.title_method_score = sentence.inner_product_title/title_self_similarity
  
  def populate_sentence_embedding(title, sentences):
    temp = [title]+sentences
    # sentence embedding for each sentence in the article including the title
    sentences_embedding = calculate_sentence_embedding(temp)
    return sentences_embedding

  def location_method(self):
    sentences = self.content
    #initialize score of 1 for title
    totalScores = [1]
    sentences[0].location_method_score = 1
    for i in range(1,len(sentences)):
      i = sentences[i].id
      j = sentences[i].para_order
      score = (1/i) * (.8* (1/j)) * .2
      sentences[i].location_method_score = score
      totalScores.append(score)
    return totalScores

  # print instance for debugging purpose
  def __repr__(self):
        return "category: {} title: {} sentences: {} sentence_embedding: {}".format(self.category, self.title, self.sentences, self.content)

def generate_summaries(categories_news):
  for category in categories_news:
    for i in range(len(categories_news[category])):
      news_article = categories_news[category][i]
      sentences = news_article.content[1:]
      # rank the sentence based on the sentence score
      sentences.sort(key = lambda x : x.sentence_score, reverse=True)
      topk = sentences[:(math.floor(len(sentences)*0.4) + 1)]
      topk.sort(key=lambda x: x.id)
      summary = topk
      file_name = (news_article.path).split("/")
      file_name[6] = "Summaries_Baseline"
      file_name[-1] = format(i+1, "03d") + ".txt"
      file_name_processed = ("/").join(file_name)
      with open(file_name_processed, "w") as file:
        for s in [sentence.content_original for sentence in summary]:
          file.write(s)
          file.write(" ")
    

def embed(input):
  return model(input) 

def calculate_sentence_embedding(contents):
  logging.set_verbosity(logging.ERROR)
  message_embeddings = embed(contents)
  return np.array(message_embeddings).tolist()

# get all the full path for all the category
def process_dirs():
  result = []
  path = pathlib.Path(__file__).resolve().parent
  subdirs = os.listdir(path)
  categories = list(filter(None, [c if re.match("^\w*$", c) else None for c in subdirs]))
  for category in categories:
    result.append(str(path) + "/" + category)
  return result, categories

def removeStopwords(sentences):
  stop_words = set(stopwords.words('english'))
  all_sentences = []
  for sentence in sentences:
    #converts sentence to words then removes stop words and puts back into sentence
    word_tokens = nltk.word_tokenize(sentence)
    filtered_sentence = [wnl.lemmatize(w) for w in word_tokens if (not w.lower() in stop_words) and (w.isalnum() or re.match("^\w+-\w+", w))]
    sentence = " ".join(filtered_sentence)
    all_sentences.append(sentence)
  return all_sentences


def removeStopwordsTitle(sentence):
  stop_words = set(stopwords.words('english'))
  #converts sentence to words then removes stop words and puts back into sentence
  word_tokens = nltk.word_tokenize(sentence)
  filtered_sentence = [wnl.lemmatize(w) for w in word_tokens if (not w.lower() in stop_words) and (w.isalnum() or re.match("^\w+-\w+", w))]
  sentence = " ".join(filtered_sentence)
  return sentence

def paraOrder(sentences):
  orders = []
  paras = sentences.split('\n\n')
  for p in paras:
    sent = nltk.sent_tokenize(p)
    for i in range(len(sent)):
      orders.append(i+1)
  return orders

# read in the document
def input_documents():
  category_paths, categories = process_dirs()
  # print("categories: "  + str(categories))
  category_news = {category:[] for category in categories}
  # process the news in each category
  for i in range(len(category_paths)):
    # get all the news article file name
    news_article_name_inorder = os.listdir(category_paths[i])
    # get the ordered news article file names list
    news_article_name_inorder = sorted(news_article_name_inorder, key=lambda x : int(x.split(".")[0]))
    # go through each news article in the category and create a list of news object for each category
    news_of_one_category = []
    for news_article in news_article_name_inorder:
      path = category_paths[i] + "/" + news_article
      # print(path)
      with open(path, 'r', errors='ignore') as file:
        # try:
        lines = file.read()
        para_order = paraOrder(lines)
        sent_text = nltk.sent_tokenize(lines)

        if ("\n\n" in sent_text[0]):
          title_with_first_sentence = sent_text[0].split("\n\n")
          # remove the stop words in the title
          title = title_with_first_sentence[0]
          # make sure sent_text only contains text sentences (no title)
          sent_text[0] = title_with_first_sentence[1]
        else:
          title = sent_text[0]
          # make sure sent_text only contains text sentences (no title)
          sent_text.pop(0)
        #to remove stopwords from an array of sentences 
        sent_text = sent_text
        #end
        sentences = sent_text
        # create News object using current news article
        news = News(categories[i], title, sentences,para_order, path)
        # print(news.location_method())
        news_of_one_category.append(news)
        # print(news)
        break
    category_news[categories[i]] = news_of_one_category
  return category_news
  
def main():
  news = input_documents()
  generate_summaries(news)

main()

