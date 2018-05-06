"""
Copyright 2013 Mark Dredze. All rights reserved.
This software is released under the 2-clause BSD license.
Mark Dredze, mdredze@cs.jhu.edu
"""
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from time import gmtime, strftime, localtime
from http.client import IncompleteRead
from utils import parseCommandLine, usage
import gzip, logging, datetime
import time
import os
import json

class FileListener(StreamListener):
    def __init__(self, path, restart_time):
        self.path = path
        self.current_file = None
        self.restart_time = restart_time
        self.file_start_time = time.time()
        self.file_start_date = datetime.datetime.now()

    def on_data(self, data):
        #print(json.loads(data))
        tmp = json.loads(data)
        print(tmp['text'])
        current_time = datetime.datetime.now()
        if self.current_file == None or time.time() - self.restart_time > self.file_start_time \
                or self.file_start_date.day != current_time.day:
            self.startFile()
            self.file_start_date = datetime.datetime.now()
        if data.startswith('{'):
            self.current_file.write(data)
            if not data.endswith('\n'):
                self.current_file.write('\n')

    def on_error(self, status):
        logger.error(status)

    def startFile(self):
        if self.current_file:
            self.current_file.close()

        local_time_obj = localtime()
        datetime = strftime("%Y_%m_%d_%H_%M_%S", local_time_obj)
        year = strftime("%Y", local_time_obj)
        month = strftime("%m", local_time_obj)

        full_path = os.path.join(self.path, year)
        full_path = os.path.join(full_path, month)
        try:
            os.makedirs(full_path)
            logger.info('Created %s' % full_path)
        except:
            pass
        filename = os.path.join(full_path, '%s.txt.gz' % datetime)
        self.current_file = gzip.open(filename, 'wt')
        self.file_start_time = time.time()
        logger.info('Starting new file: %s' % filename)


if __name__ == '__main__':

    access_token = '' # fill in your twitter API access token
    access_token_secret = '' # fill in your twitter API access token secret
    consumer_key = '' # fill in your twitter API consumer key
    consumer_secret = '' # fill in your twitter API consumer secret
    stream_type = 'keyword'
    output_directory = 'your local directory'
    track_keyword = ['trade war']
    language = ['en']
    log_filename = 'logFileKeyword.txt'

    logger = logging.getLogger('tweepy_streaming')
    handler = logging.FileHandler(log_filename, mode='a')
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # save data to file, a new file per 21600 sec, = 6 hours,
    # Be careful When you truncate the program, as the current file that data is written in will become invalid because EOF can not be reached.
    listener = FileListener(output_directory, 21600) # use 86400 for a day
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    try:
        while True:
            try:
                logger.warning("Connecting")
                stream = Stream(auth, listener)
                stream.filter(track=track_keyword, languages=language)
            except IncompleteRead as e:
                logger.error('Exception: ' + str(e))
    except Exception as e:
        logger.error('Exception: ' + str(e))
    logger.info('Exiting.')
