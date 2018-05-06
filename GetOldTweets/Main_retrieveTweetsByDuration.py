"""
Copyright 2013 Mark Dredze. All rights reserved.
This software is released under the 2-clause BSD license.
Mark Dredze, mdredze@cs.jhu.edu

This code is a hybrid of streaming_downloader and getOld-tweet

"""
from time import gmtime, strftime, localtime
from http.client import IncompleteRead
from datetime import timedelta, date
from utils import parseCommandLine, usage

import os
import json
import gzip, logging, datetime
import got

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

if __name__ == '__main__':

    keyword = 'trade war'
    start_date = date(2018, 1, 1)
    end_date = date(2018, 5, 5)

    for single_date in daterange(start_date, end_date):
        year = str(single_date.year)
        month = single_date.strftime('%m')
        search_date_begin = single_date.strftime('%Y-%m-%d')
        search_date_end = (single_date + timedelta(1)).strftime('%Y-%m-%d')
        output_directory = '/Your_local_directory/'
        full_path = os.path.join(output_directory, year)
        full_path = os.path.join(full_path, month)
        try:
            os.makedirs(full_path)
        except:
            pass

        current_file = None
        filename = os.path.join(full_path, 'keyword_tradewar_%s.txt.gz' % single_date)
        current_file = gzip.open(filename, 'wt')

        # Get tweets by query search
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(keyword).setSince(search_date_begin).setUntil(search_date_end).setMaxTweets(100000).setLang('en')
        tweet = got.manager.TweetManager.getTweets(tweetCriteria)

        for t in tweet:
            json.dump(t, current_file)
            current_file.write('\n')

        current_file.close()




