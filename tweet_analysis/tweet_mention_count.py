import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from pandas_datareader import data as web
from datetime import date, datetime, timedelta
import urllib.request, urllib.parse, urllib.error
import pandas as pd
import numpy as np
import pytz
import codecs
import csv
import re

def datetime_range_generator(start_time, end_time):
    num_hours = int((end_time - start_time).days * 24 + (end_time - start_time).seconds/3600)
    for n in range(num_hours):
        yield start_time + timedelta(hours=n)


if __name__ == '__main__':

    start_time = datetime(2018, 3, 2, 15, 0, 0)
    end_time = datetime(2018, 5, 11, 20, 0, 0)

    # to sum the number of tweets mentioning the "keyword" for every hour from 2018-03-02 15:00.

    df1 = pd.read_csv(
        '/Users/Sherry/PycharmProjects/TradeWar/0_analysis_results_files/' +
        'tweet_keyword_tradewar_combined_2_removeHTTP.csv', encoding="ISO-8859-1")
    #keyword_list = ['apple', 'soybean', 'aluminium', 'amazon', 'huawei', 'verizon', 'at&t', 'qualcomm', 'tech', 'oil', 'petro', 'yuan', 'dollar', 'telecom', 'aircraft', 'boeing', 'facebook', 'netflix', 'chip']
    keyword_list = ['intel', 'broadcom', 'acacia', 'google', 'ibm', 'tesla', 'semiconductor', 'steel', 'pork', 'solar']

    for keyword in keyword_list:
        filtered_df1 = df1[df1['Content'].str.contains(keyword, case=False, na=False)]
        stock_dates_dict = {t: 0 for t in datetime_range_generator(start_time, end_time)}

        for j, line in filtered_df1.iterrows():
            time = datetime.strptime(line[0], '%a %b %d %X +0000 %Y')
            if time.minute == 0 and time.second == 0:
                time = time.replace(second=0, minute=0)
                pass
            else:
                time = time.replace(second=0, minute=0) + timedelta(hours=1)
                #time = str(time)[:-3]
            if time in stock_dates_dict:
                stock_dates_dict[time] += 1

        datefilename = '/Users/Sherry/PycharmProjects/TradeWar/0_analysis_results_files/tweet_mention_count_' + keyword + '.csv'
        datefile = codecs.open(datefilename, 'w', 'utf-8')
        writeCSV = csv.writer(datefile, delimiter=',')
        writeCSV.writerow(['Date', 'Count'])
        for key, value in stock_dates_dict.items():
            writeCSV.writerow([key, value])
        datefile.close()
        

    # to retrieve the stock data from google finance and save into a csv, to prevent dates change if queried later.
    interval_seconds = 3600
    num_days = (date.today() - date(2018, 1, 1)).days

    #symbol_list = ['SIV', 'NATIONALUM', '399231', 'AMZN', 'AAPL', '002502', 'QCOM', 'VZ', 'T', '.IXIC', 'ONGC', 'CNYUSD', 'USDCNY', 'BA', 'FB', 'NFLX', 'C29']
    symbol_list = ['INTC', 'AVGO', 'ACIA', 'GOOGL', 'IBM', 'TSLA', 'SMI', 'STLD', 'WHGLY', 'SIRC']

    for symbol in symbol_list:
        url_string = "http://finance.google.com/finance/getprices?q={symbol}".format(symbol=symbol.upper())
        url_string += "&i={interval_seconds}&p={num_days}d&f=d,o,h,l,c,v".format(interval_seconds=interval_seconds, num_days=num_days)

        #url_string = 'https://finance.google.com/finance/getprices?i={interval_seconds}'.format(interval_seconds=interval_seconds)
        #url_string += '&f=d,o,h,l,c,v&q={symbol}'.format(symbol=symbol.upper())
        #url_string += '&startdate=Jan+01%2C+2018&enddate=Mar+06%2C+2018'

        page = urllib.request.urlopen(url_string)
        df = pd.read_csv(page, skiprows=7, sep=',', names=['DATE', 'CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME'])
        unwanted = re.compile('^TIMEZONE_OFFSET=-?[0-9]+$')
        filter = df['DATE'].str.contains(unwanted)
        df = df[~filter]
        b_dateround = df['DATE'].map(lambda dt: dt[0] == 'a')
        dateround = df[b_dateround]['DATE'].map(lambda dt: int(dt[1:]))
        df['DATE2'] = dateround
        df['DATE2'] = df['DATE2'].fillna(method='ffill')
        df['DATE3'] = df[~b_dateround]['DATE'].astype(int) * interval_seconds
        df['DATE3'] = df['DATE3'].fillna(0)
        df['DATE4'] = df['DATE2'] + df['DATE3']
        df['DATE4'] = df['DATE4'].map(lambda s: datetime.utcfromtimestamp(int(s)))
        del df['DATE']
        del df['DATE2']
        del df['DATE3']
        df = df.set_index('DATE4', verify_integrity=True)
        df.index.name = 'DATE'


        datefilename = '/Users/Sherry/PycharmProjects/TradeWar/0_analysis_results_files/stock_price_change_' + symbol + '.csv'
        datefile = codecs.open(datefilename, 'w', 'utf-8')
        writeCSV = csv.writer(datefile, delimiter=',')
        writeCSV.writerow(['DATE', 'CLOSE', 'HIGH', 'LOW', 'OPEN', 'VOLUME'])
        for i, row in df.iterrows():
            if start_time <= i < end_time:
                writeCSV.writerow([i, row.CLOSE, row.HIGH, row.LOW, row.OPEN, row.VOLUME])
        datefile.close()
        



