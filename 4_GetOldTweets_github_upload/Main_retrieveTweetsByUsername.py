"""
Copyright 2013 Mark Dredze. All rights reserved.
This software is released under the 2-clause BSD license.
Mark Dredze, mdredze@cs.jhu.edu

This code is a hybrid of streaming_downloader and getOld-tweet

"""
from time import gmtime, strftime, localtime
from http.client import IncompleteRead
from utils import parseCommandLine, usage
import gzip, logging, datetime
import time
import json
import got
import os

if __name__ == '__main__':

    user_name = 'realDonaldTrump'
    #user_name = 'POTUS'
    output_directory = '/YOUR_LOCAL_DIRECTORY/'

    # Get tweets by username
    tweetCriteria = got.manager.TweetCriteria().setUsername(user_name).setMaxTweets(38000)
    tweet = got.manager.TweetManager.getTweets(tweetCriteria)

    current_file = None
    filename = os.path.join(output_directory, 'tweets_search_%s.txt.gz' % user_name)
    current_file = gzip.open(filename, 'wt')

    for t in tweet:
        json.dump(t, current_file)
        current_file.write('\n')

    current_file.close()





