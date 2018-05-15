import json
import os
import codecs
import csv

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
# matplotlib.style.use('ggplot')

import numpy as np
from datetime import timedelta, date, datetime



def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


if __name__ == "__main__":


    # 1. Collect historical tweets: Run /GetOldTweets/Main_retrieveTweetsByDuration.py
    # 2. Collect forthcoming tweets: Run /StreamingNewTweets/Main_streamingTweetsByKeyword.py
    # 3. gunzip *.gz files, and move them to the uncompressed folders
    # 4. merge streaming tweet files

    start_date = date(2018, 4, 24)
    end_date = date(2018, 5, 11)
    directory = os.fsencode('/Users/Sherry/PycharmProjects/TradeWar/2_twitter_stream_downloader/2018/uncompressed_separated/')
    merge_directory = '/Users/Sherry/PycharmProjects/TradeWar/2_twitter_stream_downloader/2018/uncompressed_merged/'
    for single_date in daterange(start_date, end_date):
        mergefilename = merge_directory + 'keyword_tradewar_2018-' + str(single_date.strftime('%m')) + '-' + str(single_date.strftime('%d')) + '.txt'
        tweet_count = 0
        for file in sorted(os.listdir(directory)):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                if filename.find('2018_' + str(single_date.strftime('%m')) + '_' + str(single_date.strftime('%d'))) != -1:
                    with codecs.open(mergefilename, "a", 'utf-8') as mergefile:
                        tweets_filename = os.path.join(directory, file)
                        tweets_file = codecs.open(tweets_filename, "r", 'utf-8')
                        for line in tweets_file:
                            mergefile.write(line)
                            tweet_count += 1
        print(tweet_count)

    # 5. move the merged files to the same folder as oldTweets .txt files.
    # 6. run the code below to extract tweet contents and the time they were posted from the json files, and combine into a single file "tweet_keyword_tradewar_combined.csv".

    # specify outputfile name
    outputfile = '/Users/Sherry/PycharmProjects/TradeWar/0_analysis_results_files/tweet_keyword_tradewar_combined_2.csv'
    outputfile_count = '/Users/Sherry/PycharmProjects/TradeWar/0_analysis_results_files/tweet_daily_count_2.csv'

    # process historical + merged_streaming tweets
    directory = os.fsencode('/Users/Sherry/PycharmProjects/TradeWar/1_GetOldTweets-python/2018/uncompressed/')
    outputf = codecs.open(outputfile, 'w', 'utf-8')
    csvwriter = csv.writer(outputf, delimiter=',')
    csvwriter.writerow(['Date', 'Content'])
    
    outputf_count = codecs.open(outputfile_count, 'w', 'utf-8')
    csvwriter_count = csv.writer(outputf_count, delimiter=',')
    csvwriter_count.writerow(['Date', 'Count'])

    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            tweets_filename = os.path.join(directory, file)
            tweets_file = codecs.open(tweets_filename, "r", 'utf-8')
            tweet_count = 0
            for line in tweets_file:
                tweet = json.loads(line.strip())
                if 'formatted_date' in tweet:
                    tdate = tweet['formatted_date']
                    txt = tweet['text']
                    txt = txt.replace('\n', ' ')
                    csvwriter.writerow([tdate, txt])
                    tweet_count += 1
                elif 'created_at' in tweet:
                    tdate = tweet['created_at']
                    if 'retweeted_status' in tweet and tweet['retweeted_status']['truncated']:
                        txt = tweet['retweeted_status']['extended_tweet']['full_text']
                    elif tweet['truncated']:
                        txt = tweet['extended_tweet']['full_text']
                    else:
                        txt = tweet['text']
                    txt = txt.replace('\n', ' ')
                    csvwriter.writerow([tdate, txt])
                    tweet_count += 1
            csvwriter_count.writerow([datetime.strptime(tdate, '%a %b %d %X +0000 %Y').strftime('%Y-%m-%d'), tweet_count])
            tweets_file.close()
    outputf.close()
    outputf_count.close()


    # 7. alternatively, run the code below to remove HTTP links.
    # specify outputfile name

    outputfile = '/Users/Sherry/PycharmProjects/TradeWar/0_analysis_results_files/tweet_keyword_tradewar_combined_2_removeHTTP.csv'
    outputfile_count = '/Users/Sherry/PycharmProjects/TradeWar/0_analysis_results_files/tweet_daily_count_2_removeHTTP.csv'

    # process historical + merged_streaming tweets
    directory = os.fsencode('/Users/Sherry/PycharmProjects/TradeWar/1_GetOldTweets-python/2018/uncompressed/')
    outputf = codecs.open(outputfile, 'w', 'utf-8')
    csvwriter = csv.writer(outputf, delimiter=',')
    csvwriter.writerow(['Date', 'Content'])

    outputf_count = codecs.open(outputfile_count, 'w', 'utf-8')
    csvwriter_count = csv.writer(outputf_count, delimiter=',')
    csvwriter_count.writerow(['Date', 'Count'])

    for file in sorted(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            tweets_filename = os.path.join(directory, file)
            tweets_file = codecs.open(tweets_filename, "r", 'utf-8')
            tweet_count = 0
            for line in tweets_file:
                tweet = json.loads(line.strip())
                if 'formatted_date' in tweet:
                    tdate = tweet['formatted_date']
                    txt = tweet['text']
                elif 'created_at' in tweet:
                    tdate = tweet['created_at']
                    if 'retweeted_status' in tweet and tweet['retweeted_status']['truncated']:
                        txt = tweet['retweeted_status']['extended_tweet']['full_text']
                    elif tweet['truncated']:
                        txt = tweet['extended_tweet']['full_text']
                    else:
                        txt = tweet['text']

                #txt = re.sub("[^a-zA-Z]", " ", txt)  # Removing numbers and punctuation
                txt = txt.replace('\n', ' ')
                #txt = re.sub(" +", " ", txt)  # Removing extra white space
                stop_idx = txt.find('http')
                if stop_idx != -1:
                    txt = txt[:stop_idx]
                csvwriter.writerow([tdate, txt])
                tweet_count += 1

            csvwriter_count.writerow([datetime.strptime(tdate, '%a %b %d %X +0000 %Y').strftime('%Y-%m-%d'), tweet_count])
            tweets_file.close()
    outputf.close()
    outputf_count.close()





