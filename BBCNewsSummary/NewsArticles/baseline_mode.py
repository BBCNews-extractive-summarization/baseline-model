#@title Load the Universal Sentence Encoder's TF Hub module
from __future__ import print_function
from ftplib import all_errors
from lib2to3.pgen2 import tokenize
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import pathlib
import seaborn as sns
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')

nltk.download('stopwords')
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)

class Sentences:
      def __init__(self, content, id, sentence_embedding, para_order):
            self.id = id
            self.content = content
            self.sentence_embedding = sentence_embedding
            self.para_order = para_order
            self.inner_product_title = None

      # print instance for debugging purpose
      def __repr__(self):
            # !!! for now just show the first three sentence embedding for the purpose of reduce the computing power used
            return "id: {} content: {} sentence_embedding: {}".format(self.id, self.content, self.sentence_embedding[:3])


class News:
  def __init__(self, category = None, title = None, sentences = None, para_order = None):
    self.category = category
    self.sentences = sentences
    self.para_order = para_order
    # sentences, index corresponding to sentence id. eg: title has index 0, id 0
    self.content = []
    sentences_embedding = News.populate_sentence_embedding(title, sentences)
    for i, embedding in enumerate(sentences_embedding):
        if (i == 0):
          self.title = Sentences(title, i, embedding, 1)
          self.content.append(self.title)
        else:
          self.content.append(Sentences(sentences[i-1], i, embedding, para_order[i]))
    self.run_title_method()

  def run_title_method(self):
    # calcluate the similarity of title to itself for later normalization
    title_self_similarity = np.inner(self.title.sentence_embedding, self.title.sentence_embedding)
    for sentence in self.content:
      # calculate the similarity of title and each sentence
      sentence.inner_product_title = np.inner(self.title.sentence_embedding, sentence.sentence_embedding)
      # normalize to the range of 0-1
      sentence.inner_product_title = sentence.inner_product_title/title_self_similarity
  
  def populate_sentence_embedding(title, sentences):
    temp = [title]+sentences
    # sentence embedding for each sentence in the article including the title
    sentences_embedding = calculate_sentence_embedding(temp)
    return sentences_embedding

  def location_method(self):
    sentences = self.content
    #initialize score of 1 for title
    totalScores = [1]
    for i in range(1,len(sentences)):
      i = sentences[i].id
      j = sentences[i].para_order
      score = (1/i) * (.8* (1/j)) * .2
      totalScores.append(score)
    return totalScores

  # print instance for debugging purpose
  def __repr__(self):
        return "category: {} title: {} sentences: {} sentence_embedding: {}".format(self.category, self.title, self.sentences, self.content)

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
    filtered_sentence = [w for w in word_tokens if (not w.lower() in stop_words) and (w.isalnum() or re.match("^\w+-\w+", w))]
    sentence = " ".join(filtered_sentence)
    all_sentences.append(sentence)
  return all_sentences


def removeStopwordsTitle(sentence):
  stop_words = set(stopwords.words('english'))
  #converts sentence to words then removes stop words and puts back into sentence
  word_tokens = nltk.word_tokenize(sentence)
  filtered_sentence = [w for w in word_tokens if (not w.lower() in stop_words) and (w.isalnum() or re.match("^\w+-\w+", w))]
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
          title = removeStopwordsTitle(title_with_first_sentence[0])
          # make sure sent_text only contains text sentences (no title)
          sent_text[0] = title_with_first_sentence[1]
        else:
          title = removeStopwordsTitle(sent_text[0])
          # make sure sent_text only contains text sentences (no title)
          sent_text.pop(0)
        #to remove stopwords from an array of sentences 
        sent_text = removeStopwords(sent_text)
        #end
        sentences = sent_text
        # create News object using current news article
        news = News(categories[i], title, sentences,para_order)
        # print(news.location_method())
        news_of_one_category.append(news)
    category_news[categories[i]] = news_of_one_category
  return category_news
  
def main():
  input_documents()

main()

# # this is for plotting similarity between sentences (eg: title and sentences, sentences and sentences) to be used in the paper
# def plot_similarity(labels, features, rotation):
#   corr = np.inner(features, features)
#   sns.set(font_scale=1.2)
#   g = sns.heatmap(
#       corr,
#       xticklabels=labels,
#       yticklabels=labels,
#       vmin=0,
#       vmax=1,
#       cmap="YlOrRd")
#   g.set_xticklabels(labels, rotation=rotation)
#   g.set_title("Semantic Textual Similarity")

# def run_and_plot(messages_):
#   message_embeddings_ = embed(messages_)
#   plot_similarity(messages_, message_embeddings_, 90)

# messages = [
#     # Smartphones
#     "I like my phone",
#     "My phone is not good.",
#     "Your cellphone looks great.",

#     # Weather
#     "Will it snow tomorrow?",
#     "Recently a lot of hurricanes have hit the US",
#     "Global warming is real",

#     # Food and health
#     "An apple a day, keeps the doctors away",
#     "Eating strawberries is healthy",
#     "Is paleo better than keto?",

#     # Asking about age
#     "How old are you?",
#     "what is your age?",
# ]

# run_and_plot(messages)
               




# # this part is to get the similarity between sentence embedding scores
# import pandas
# import scipy
# import math
# import csv

# sts_dataset = tf.keras.utils.get_file(
#     fname="Stsbenchmark.tar.gz",
#     origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
#     extract=True)
# sts_dev = pandas.read_table(
#     os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-dev.csv"),
#     error_bad_lines=False,
#     skip_blank_lines=True,
#     usecols=[4, 5, 6],
#     names=["sim", "sent_1", "sent_2"])
# sts_test = pandas.read_table(
#     os.path.join(
#         os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"),
#     error_bad_lines=False,
#     quoting=csv.QUOTE_NONE,
#     skip_blank_lines=True,
#     usecols=[4, 5, 6],
#     names=["sim", "sent_1", "sent_2"])
# # cleanup some NaN values in sts_dev
# sts_dev = sts_dev[[isinstance(s, str) for s in sts_dev['sent_2']]]




# sts_data = sts_dev

# def run_sts_benchmark(batch):
#   sts_encode1 = tf.nn.l2_normalize(embed(tf.constant(batch['sent_1'].tolist())), axis=1)
#   sts_encode2 = tf.nn.l2_normalize(embed(tf.constant(batch['sent_2'].tolist())), axis=1)
#   cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
#   clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
#   scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
#   """Returns the similarity scores"""
#   return scores

# dev_scores = sts_data['sim'].tolist()
# scores = []
# for batch in np.array_split(sts_data, 10):
#   scores.extend(run_sts_benchmark(batch))

# pearson_correlation = scipy.stats.pearsonr(scores, dev_scores)
# print('Pearson correlation coefficient = {0}\np-value = {1}'.format(
#     pearson_correlation[0], pearson_correlation[1]))
