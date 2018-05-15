import json
import os
import codecs
import csv

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
#matplotlib.style.use('ggplot')

import re, nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora, models
import gensim
import numpy as np
from datetime import timedelta, date, datetime
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation




stemmer = PorterStemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

lemmatizer = WordNetLemmatizer()
def lemmatize_tokens(tokens, lemmatizer):
    lemmatized = []
    for item in tokens:
        lemmatized.append(lemmatizer.lemmatize(item))
    return lemmatized


# nltk.download('wordnet')
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in lda_tokenize(articles)]

def tokenize(text):
    text = re.sub("[^a-zA-Z]", " ", text) # Removing numbers and punctuation
    text = re.sub(" +"," ", text) # Removing extra white space
    text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b"," ",text) # Removing very long words above 10 characters
    text = re.sub("\\b[a-zA-Z0-9]{0,1}\\b"," ",text) # Removing single characters (e.g k, K)
    tokens = nltk.word_tokenize(text.strip())
    # use stemmer
    # tokens = stem_tokens(tokens, stemmer)
    # use lemmatizer
    tokens = lemmatize_tokens(tokens, lemmatizer)
    tokens = nltk.pos_tag(tokens)

    return tokens

def lda_tokenize(text):
    text = re.sub("[^a-zA-Z]", " ", text) # Removing numbers and punctuation
    text = re.sub(" +"," ", text) # Removing extra white space
    text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b"," ",text) # Removing very long words above 10 characters
    # text = re.sub("\\b[a-zA-Z0-9]{0,1}\\b"," ",text) # Removing single characters (e.g k, K)
    tokens = nltk.word_tokenize(text.strip())
    return tokens

def preprocess(fileObj, Uname):

    readCSV = csv.reader(fileObj, delimiter=',')
    stemfilename = 'tweet_keyword_tradewar_combined_stem.csv'
    stemfile = codecs.open(stemfilename, 'w', 'utf-8')
    writeCSV = csv.writer(stemfile, delimiter=',')

    text_corpus = []
    for doc in readCSV:
        temp_doc = tokenize(doc[0].strip())
        current_doc = []
        for word in range(len(temp_doc)):
            if temp_doc[word][0] not in stop_words and temp_doc[word][1][:2] in ['NN', 'JJ', 'PR', 'VB']: # 'NNS', 'NNP', 'NNPS',
                current_doc.append(temp_doc[word][0])
        #text_corpus.append(current_doc)
        writeCSV.writerow(current_doc)
    stemfile.close()


def initialize_stop_words():
    stop_words = stopwords.words('english')
    stop_words.extend(['this', 'that', 'the', 'might', 'have', 'been', 'from',
                       'but', 'they', 'will', 'has', 'having', 'had', 'how', 'went', 'were', 'why', 'and', 'still', 'his',
                       'her', 'was', 'its', 'per', 'cent', 'a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among',
                       'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can',
                       'cannot', 'could', 'dear', 'did', 'do', 'does', 'either', 'else', 'ever', 'every',
                       'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his',
                       'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let',
                       'like', 'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'nor',
                       'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said',
                       'say', 'says', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their',
                       'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'want',
                       'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who',
                       'whom', 'why', 'will', 'with', 'would', 'yet', 'you', 'your', 've', 're', 'rt', 'retweet',
                       'fuckem', 'fuck', 'fuck', 'ya', 'yall', 'yay', 'youre', 'youve', 'ass', 'factbox', 'com', 'lt', 'th',
                       'retweeting', 'dick', 'fuckin', 'shit', 'via', 'fucking', 'shocker', 'wtf', 'hey', 'ooh',
                       'rt&amp', 'http', 'https', 'amp', 'com', 'co', 'html', 'www', 'äî', 'äô', 'äôre', 'äôs', 'äôt', 'äù',
                       'babas', 'anos', 'yvdypkd', 'twitter', 'donald', 'realdonaldtrump', 'shr', 'gs', 'yvdypkb', 'start', 'news', 'bloomberg', 'article',
                       'cbc', 'ca', 'msn', 'look', 'cnn', 'canucks', 'bbjee', 'cmp', 'btn', 'uswarcriminals', 'blogspot', 'youtu', 'n pr hz rb',
                       'washin', 'gtons', ' spref', 'tw', 'tid', 'goo', 'gl', 'spr', 'ly', 'times', 'nyt', 'nytimes', 'nyti', 'iduskbn', 'hmmm', 'tgs',
                       'rss', 'aavzs', 'lf', 'reut', 'rs', 'okjzess', 'qoorn', 'reviveoldpost', 'axsoq', 'ukwyg', 'hwxn', 'ipmy', 'cnnmoney', 'hgwwg', ' co', 'jp',
                       'hgwwgycojp', 'imojdyjb', 'yt', 'nzhgpsi', 'washpostbiz', 'mx', 'kgtmlah', 'ogcwejd', 'ha', 'na', 'ift', 'tt', 'feedproxy', 'zerohedge'])
                        # cannot remove 2-gram from here, need to filter the saved csv.
    return stop_words

def splitTweet(fileObj, begin_date, end_date, output_dir):

    readCSV = csv.reader(fileObj, delimiter=',')

    splitFilefolder = str(begin_date) + '_to_' +str(end_date)
    splitFilepath = os.path.join(output_dir, splitFilefolder)
    try:
        os.makedirs(splitFilepath)
    except:
        pass
    splitFilename = splitFilepath + '/tweet_keyword_tradewar_combined_split_' + str(begin_date) + '_to_' +str(end_date) + '.csv'
    splitfile = codecs.open(splitFilename, 'w', 'utf-8')
    writeCSV = csv.writer(splitfile, delimiter=',')

    for line in readCSV:
        tweet_date = line[0].strip()
        tweet_date = datetime.strptime(tweet_date, '%a %b %d %X +0000 %Y')
        if end_date > datetime.date(tweet_date) >= begin_date:
            writeCSV.writerow(line)
    splitfile.close()

def splitTweet_removeDate(fileObj, begin_date, end_date, output_dir):

    readCSV = csv.reader(fileObj, delimiter=',')

    splitFilefolder = str(begin_date) + '_to_' +str(end_date)
    splitFilepath = os.path.join(output_dir, splitFilefolder)
    try:
        os.makedirs(splitFilepath)
    except:
        pass
    splitFilename = splitFilepath + '/tweet_keyword_tradewar_combined_split_' + str(begin_date) + '_to_' +str(end_date) + '.csv'
    splitfile = codecs.open(splitFilename, 'w', 'utf-8')
    writeCSV = csv.writer(splitfile, delimiter=',')

    for line in readCSV:
        tweet_date = line[0].strip()
        tweet_date = datetime.strptime(tweet_date, '%a %b %d %X +0000 %Y')
        if end_date > datetime.date(tweet_date) >= begin_date:
            writeCSV.writerow([line[1].strip()])
    splitfile.close()


def splitTweet_removeDate_removeHTTPRetweet(fileObj, begin_date, end_date, output_dir):
    readCSV = csv.reader(fileObj, delimiter=',')

    splitFilefolder = str(begin_date) + '_to_' + str(end_date)
    splitFilepath = os.path.join(output_dir, splitFilefolder)
    try:
        os.makedirs(splitFilepath)
    except:
        pass
    splitFilename = splitFilepath + '/tweet_keyword_tradewar_combined_split_' + str(begin_date) + '_to_' + str(
        end_date) + '_rmHTTPRetweet.csv'
    splitfile = codecs.open(splitFilename, 'w', 'utf-8')
    writeCSV = csv.writer(splitfile, delimiter=',')

    tweet_dictionary = {}
    for line in readCSV:
        tweet_date = line[0].strip()
        tweet_date = datetime.strptime(tweet_date, '%a %b %d %X +0000 %Y')
        if end_date > datetime.date(tweet_date) >= begin_date:
            line_to_write = line[1].strip()
            stop_idx = line_to_write.find('http')
            if stop_idx != -1:
                line_to_write = line_to_write[:stop_idx]

            line_to_write = re.sub("[^a-zA-Z]", " ", line_to_write)  # Removing numbers and punctuation
            line_to_write = re.sub(" +", " ", line_to_write)  # Removing extra white space
            #line_to_write = " ".join(line_to_write)

            if line_to_write in tweet_dictionary:
                pass
            elif len(line_to_write)>0:
                writeCSV.writerow([line_to_write])
                tweet_dictionary[line_to_write] = 1
    splitfile.close()


def wordcloud_in_one_figure(outputPath, num_features, df_gradients, version):

    figureName = outputPath + 'figure_tweet_wordCloud.png'
    plt.rcParams['figure.figsize'] = (15.0, 15.0)
    fig, ax = plt.subplots(3, 3)

    count = 0
    for max_fq in df_gradients:
        tfidffilename = outputPath + 'tweet_keyword_tradewar_tfidf_features_' + str(max_fq) + '_' + str(
            num_features) + '_v' + str(version) + '.csv'
        word_text = open(tfidffilename).read()
        wordcloud = WordCloud(colormap='hsv', max_font_size=90, max_words=800, width=1000, height=1000,
                              margin=3, collocations=False).generate(word_text)
        axes = ax[count % 3, count // 3]
        axes.imshow(wordcloud, interpolation="bilinear")
        axes.set_xlabel('max_df = ' + str(max_fq))
        axes.axis("off")
        count += 1
        tfidffilename.close()

    fig.savefig(figureName, dpi=90)

def saveTerms_sortedTFIDFscores(outputPath, max_fq, num_features, version, tfidf_feature_names, tfidf_matrix):
    tfidffilename = outputPath + 'tweet_keyword_tradewar_tfidf_features_' + str(max_fq) + '_' + str(num_features) + '_v' + str(version) + '.csv'
    tfidffile = codecs.open(tfidffilename, 'w', 'utf-8')
    writeCSV = csv.writer(tfidffile, delimiter=',')

    # features ranked by scores, list saved for wordcloud plotting
    scores = zip(tfidf_feature_names, np.asarray(tfidf_matrix.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for feature, score in sorted_scores:
        feature = feature.replace(' ', '_') # connect the words in multi-grams
        writeCSV.writerow([feature])
    tfidffile.close()

def saveTerms_sortedTFscores(outputPath, max_fq, num_features, version, tf_feature_names, tf_matrix):
    tffilename = outputPath + 'tweet_keyword_tradewar_tf_features_' + str(max_fq) + '_' + str(num_features) + '_v' + str(version) + '.csv'
    tffile = codecs.open(tffilename, 'w', 'utf-8')
    writeCSV = csv.writer(tffile, delimiter=',')

    scores = zip(tf_feature_names, np.asarray(tf_matrix.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for feature, score in sorted_scores:
        feature = feature.replace(' ', '_')
        writeCSV.writerow([feature])
    tffile.close()

def display_topics(model, feature_names, num_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        #print(" ".join([feature_names[i].replace(' ', '_') for i in topic.argsort()[:-num_top_words - 1:-1]]))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]))


def display_topics_rmOverlap(model, feature_names, num_top_words, keywords_history):
    output_words = []
    for topic_idx, topic in enumerate(model.components_):
        topic_keywords = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
        output_words.extend(topic_keywords)
        topic_keywords = [word for sentence in topic_keywords for word in sentence.split()]
        output_words.extend(topic_keywords)
        if not set(topic_keywords).issubset(keywords_history):
            print("Topic %d:" % (topic_idx))
            #print(" ".join([feature_names[i].replace(' ', '_') for i in topic.argsort()[:-num_top_words - 1:-1]]))
            print(" ".join(topic_keywords))
            keywords_history.update(topic_keywords)
    return output_words


if __name__ == "__main__":


    # 1. Collect historical tweets: Run /GetOldTweets/Main_retrieveTweetsByDuration.py
    # 2. Collect forthcoming tweets: Run /StreamingNewTweets/Main_streamingTweetsByKeyword.py
    # 3. run the tweets_combine.py to merge all tweets in one csv file: "tweet_keyword_tradewar_combined.csv"
    # 4. (optional) run the tweet_mention_count.py to generate results to be used later in the market backtesting analysis.

    # generate file handler and pass to function splitTweets()
    outputFilename = 'tweet_keyword_tradewar_combined.csv'
    output_dir = '/Users/Sherry/PycharmProjects/TradeWar/0_analysis_results_files/';
    outputfile = output_dir + outputFilename
    # tweetTradeWar = codecs.open(outputfile, 'r', 'utf-8')

    # to stem/lemmatize the words (optional)
    # preprocess(tweetTradeWar, 'Trade War')

    # split the tweets based on dates, and then remove the date column for keyword analysis
    begin_dates = [date(2018, 1, 1),  date(2018, 3, 1), date(2018, 3, 22), date(2018, 4, 1)]
    end_dates = [date(2018, 3, 1),  date(2018, 3, 22), date(2018, 4, 1), date(2018, 5, 4)]
    for idate in range(4):
        begin_date = begin_dates[idate]
        end_date = end_dates[idate]
        tweetTradeWar = codecs.open(outputfile, 'r', 'utf-8')
        #splitTweet_removeDate(tweetTradeWar, begin_date, end_date, output_dir)
        splitTweet_removeDate_removeHTTPRetweet(tweetTradeWar, begin_date, end_date, output_dir)
        tweetTradeWar.close()

    # import the split tweet file one by one, and do analysis
    output_dir = '/Users/Sherry/PycharmProjects/TradeWar/0_analysis_results_files'

    begin_dates = [date(2018, 1, 1), date(2018, 3, 1), date(2018, 3, 22), date(2018, 4, 1)]
    end_dates = [date(2018, 3, 1), date(2018, 3, 22), date(2018, 4, 1), date(2018, 5, 4)]
    for idate in range(1):
        begin_date = begin_dates[idate]
        end_date = end_dates[idate]
        outputPath = output_dir + '/' + str(begin_date) + '_to_' +str(end_date) + '/'
        importfilename = outputPath + 'tweet_keyword_tradewar_combined_split_' + str(begin_date) + '_to_' + str(end_date) + '_rmHTTPRetweet.csv'

        num_features = 5000 #1000
        ngram_min = 4
        ngram_max = 4
        num_topics = 20
        num_top_words = 1

        df_gradients = [1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001]
        # [1.0, 0.1, 0.01, 0.001, 0.0001]

        # bag of words analysis, define the initial set of stopwords
        stop_words = initialize_stop_words()

        # tf-idf
        for max_fq in df_gradients:
            tweetImport = codecs.open(importfilename, 'r', 'utf-8')
            # NMF can use tf-idf # lowercase=False
            tfidf_vectorizer = TfidfVectorizer(strip_accents='ascii', ngram_range=(ngram_min, ngram_max), max_df=max_fq, min_df=1, max_features=num_features, stop_words=stop_words, analyzer='word', token_pattern='[a-zA-Z]+')
            tfidf_matrix = tfidf_vectorizer.fit_transform(tweetImport)
            tfidf_feature_names = tfidf_vectorizer.get_feature_names()
            stop_words.extend(tfidf_feature_names)
            tweetImport.close()

            # save the terms ranked by tfidf scores into a list, to be used for wordcloud plotting
            version = 3
            saveTerms_sortedTFIDFscores(outputPath, max_fq, num_features, version, tfidf_feature_names, tfidf_matrix)

            # Run NMF (results not as good as LDA)
            nmf = NMF(n_components=num_topics, random_state=1, alpha=0, init='random').fit(tfidf_matrix)
            display_topics(nmf, tfidf_feature_names, num_top_words)


        # plot all wordclouds in one figure
        # wordcloud_in_one_figure(outputPath, num_features, df_gradients)

        # plot individual wordclouds:
        plt.rcParams['figure.figsize'] = (10.0, 7.0)
        
        for max_fq in df_gradients:
            tfidffilename = outputPath + 'tweet_keyword_tradewar_tfidf_features_' + str(max_fq) + '_' + str(num_features) + '_v3.csv'
            tfidffile = open(tfidffilename, 'r')
            word_text = tfidffile.read()

            wordcloud = WordCloud(colormap='hsv', max_words=1000, width=3000, height=2000, margin=3, collocations=False).generate(word_text)

            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            figureName = outputPath + 'figure_tweet_wordCloud_' + str(max_fq) + '_' + str(num_features) + '.png'
            plt.savefig(figureName, dpi=90)
            tfidffile.close()


        # bag of words analysis, define the initial set of stopwords
        stop_words = initialize_stop_words()
        keywords_history = set()

        # raw term counts
        for max_fq in df_gradients:
            tweetImport = codecs.open(importfilename, 'r', 'utf-8')
            # LDA can only use raw term counts, because it is a probabilistic graphical model
            tf_vectorizer = CountVectorizer(strip_accents='ascii', ngram_range=(ngram_min, ngram_max), max_df=max_fq, min_df=1, max_features=num_features, stop_words=stop_words, token_pattern='[a-zA-Z]+')
            tf_matrix = tf_vectorizer.fit_transform(tweetImport)
            tf_feature_names = tf_vectorizer.get_feature_names()
            tweetImport.close()

            # Run LDA
            lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online', learning_offset=10., random_state=0)
            lda_result = lda.fit(tf_matrix)
            new_stopwords = display_topics_rmOverlap(lda_result, tf_feature_names, num_top_words, keywords_history)
            stop_words.extend(new_stopwords)



        


