import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, Event, State
import pandas as pd
from datetime import date, datetime, timedelta
import urllib.request, urllib.parse, urllib.error
import plotly.graph_objs as go
import visdcc
from plotly import tools
import numpy as np
import codecs
import csv
import re
import base64



app = dash.Dash()
server = app.server
#app.scripts.config.serve_locally = True

def transform_date(date_idx):
    dates = [date(2018, 1, 1), date(2018, 3, 1), date(2018, 3, 22), date(2018, 4, 1)]
    return dates[date_idx]

def transform_maxdf(maxdf_idx):
    return 10.0**(maxdf_idx-4)

my_css_urls = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
for my_css_url in my_css_urls:
    app.css.append_css({
        "external_url": my_css_url
    })

image_filename = 'data/figure1_tweet_marketIndex_small.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


markdown_text = ['Years of trade tensions between the world\'s two biggest economies have taken a worse turn since the start of 2018. ' \
                'President Donald Trump\'s "trade war" officially began on March 1 when he announced steep tariffs on imported steel and aluminium from China. '\
                'The situation escalated on March 22, when U.S. Trade Representative proposed 25 percent duties on Chinese products and China fought back by levying tariffs on $3 billion of U.S. imports. ' \
                'On April 03, the focus shifted to China\'s alleged abuse of IP and many tech companies were caught in the crossfire.',
                 '"Trade war" has become a focal topic on social media. Twitter has recorded hundreds of thousands of people\'s opinions on this topic. The figure below depicts the number of tweets generated each day including the keyword "trade war". ' \
                'The three peaks clearly match with the critical dates in the timeline mentioned above.',
                'Four panels below plotted together the market index daily close values (red) and the daily count of tweets mentioning "trade war" (blue), for four different market indexes. ' \
                'S&P 500, Dow Jones and Nasdaq are U.S. market indexes. SSE composite is a market index of stocks traded at the Shanghai Stock Exchange. A negative correlation is observed (-0.37 ~ -0.48), reflecting investors\' lack of confidence in stock market during a "trade war' \
                'The impact on market on March 01 is not as significant as on March 22 and April 03, because actions were taken into effect only on the latter two timepoints.',
                'We can zoom in to see the temporal correlation of tweet discussion around a perticular industry and the corresponding stock behavior. Stock/commodity price change was calculated for every hour during the market opening and between the close value of previous day and the close value of the first hour of the current day for market closed period (blue). Number of tweets mentioning the particular keywords (e.g. soybean, qualcomm, dollar) were summed for every hour (orange).',
                'Frequent words (uni-gram and bi-gram) co-mentioned with "trade war" in tweets were ranked by TFIDF scores. A maximum document frequency was set to uncover less frequent but critical words. Words appeared at higher document frequencies were added to the list of stopwords for subsequent wordcloud generation at lower document frequencies. ' \
                'As the capped document frequency reduces, more interesting words appear, such as "nuclear", "brussels", "belt", "stoking". \n ',
                'One step further, topics mentioned in these tweets were extracted by Latent Dirichlet Allocation. Four-gram topics were generated to best capture topic meanings. At each document frequency, three rounds of LDA were conducted to retrieve in total 30 topics. The use of stopwords follows the approach in wordcloud section. \n',
                'Maximum document frequency:']


df0 = pd.read_csv(
    'data/'+
    'tweet_daily_count_2.csv')


app.layout = html.Div([

    html.Div([
        html.H1(children='Understanding "Trade War" from Twitter'),
        dcc.Markdown(children=markdown_text[0]),
    ], style={'marginLeft': '5%', 'marginRight': '5%'}),
    html.Br(),
    html.Div([
        html.H3(children='1. Daily count of tweets mentioning "trade war"'),
        dcc.Markdown(children=markdown_text[1]),
        html.Div([
            dcc.Graph(
            id='tweets_count',
            figure={
                'data': [
                    go.Scatter(
                        x=df0['Date'],
                        y=df0['Count'],
                        #text=df0['Date'],
                        mode='lines',
                        name= 'Daily tweet count'
                    )
                ],
                'layout': go.Layout(
                    #xaxis={'title': 'Date'},
                    yaxis={'title': 'Num tweets containing "trade war"'},
                    margin={'l': 150, 'b': 100, 't': 50, 'r': 150},
                    #legend={'x': 0, 'y': 1},
                    hovermode='closest'
                    )
                }
            ),
        ]),
    ], style={'marginLeft': '5%', 'marginRight': '5%'}),
    html.Div([
        html.H3(children='2. Market indexes slump when "trade war" discussion heats up on Twitter'),
        dcc.Markdown(children=markdown_text[2]),
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
        ], style={'width': '80%', 'align': 'left'})
    ], style={'marginLeft': '5%', 'marginRight': '5%'}),
    #html.Br(),
    html.Div([
        html.H3(children='3. Market Impact Prediction Power'),
        dcc.Markdown(children=markdown_text[3]),
        html.Br(),
        html.Div([
            dcc.Dropdown(
            id='dropdown1',
            options=[
                {'label': 'Soybean', 'value': 'SIV'},
                {'label': 'Dollar', 'value': 'USDCNY'},
                #{'label': 'Aluminium', 'value': 'NATIONALUM'},
                #{'label': 'Steel', 'value': 'NATIONALUM'},
                #{'label': 'Agriculture Index', 'value': '399231'},
                #{'label': 'Boeing', 'value': 'BA'},
                #{'label': 'Oil&Gas', 'value': 'ONGC'},
                #{'label': 'Amazon', 'value': 'AMZN'},
                #{'label': 'Apple', 'value': 'AAPL'},
                {'label': 'Qualcomm', 'value': 'QCOM'},
                #{'label': 'Verizon', 'value': 'VZ'},
                #{'label': 'AT&T', 'value': 'T'},
                #{'label': 'Chip', 'value': 'C29'},
                #{'label': 'Netflix', 'value': 'NFLX'},
                #{'label': 'Facebook', 'value': 'FB'},
                #{'label': 'Tech', 'value': '.IXIC'},
                #{'label': 'Yuan', 'value': 'CNYUSD'},
                #{'label': 'Intel', 'value': 'INTC'},
                #{'label': 'YEN', 'value': 'USDJPY'},
                {'label': 'Broadcom', 'value': 'AVGO'},
                #{'label': 'Acacia', 'value': 'ACIA'},
                {'label': 'Google', 'value': 'GOOGL'},
                #{'label': 'IBM', 'value': 'IBM'},
                #{'label': 'Tesla', 'value': 'TSLA'},
                #{'label': 'Semiconductor', 'value': 'SMI'},
                #{'label': 'Steel', 'value': 'STLD'},
                #{'label': 'Pork', 'value': 'WHGLY'},
                #{'label': 'Solar panel', 'value': 'SIRC'},

            ],
            value='SIV'
        )], style={
            'width': '80%',
            'height': 50,
            'margin-left': 'auto',
            'margin-right': 'auto'
        }),
        html.Div([
            dcc.Graph(id='stock_price_and_tweet_mention'),
        ], style={
            'width': '95%',
            #'height': 400,
            'margin-left': 'auto',
            'margin-right': 'auto'
        }),
    ], style={'marginLeft': 50, 'marginRight': 50}),
    html.Br(),
    html.Br(),
    html.Div([
        html.H3(children='4. Wordcloud summary'),
        dcc.Markdown(children=markdown_text[4]),
        html.Div([
            dcc.Tabs(
                    tabs=[
                        {'label': 'Jan 01 - Feb 28', 'value': 0},
                        {'label': 'Mar 01 - Mar 21', 'value': 1},
                        {'label': 'Mar 22 - Mar 31', 'value': 2},
                        {'label': 'Apr 01 - May 04', 'value': 3}
                    ],
                    value=0,
                    id='tabs1'
                ),
        ], style={
            'width': '80%',
            'fontFamily': 'Sans-Serif',
            'height': 60,
            'margin-left': 'auto',
            'margin-right': 'auto',
            'float': 'center'
        }),
        html.Div([
        dcc.Markdown(children=markdown_text[6]),
        dcc.Slider(
            id='maxdf-slider1',
            marks={i: '{}'.format(10.0**(i-4)) for i in range(5)},
            min=0,
            max=4,
            value=4,
            updatemode='drag',
            #vertical = True
        )], style={
            'width': '80%',
            'height': 20,
            'margin-left': 'auto',
            'margin-right': 'auto',
            'float': 'center'
            }),
        html.Div([
            html.Img(id='wordcloud')
        ], style={
            'width': 200,
            'height': 600,
            #'align': 'middle',
            #'display': 'inline-block',
            'margin-left': '7%',
            'margin-right': '7%',
        }),
    ], style={'marginLeft': '5%', 'marginRight': '5%'}),
    html.Br(),
    html.Br(),
    html.Div([
        html.H3(children='5. LDA Topic Analysis'),
        dcc.Markdown(children=markdown_text[5]),
        html.Div([
            dcc.Tabs(
                    tabs=[
                        {'label': 'Jan 01 - Feb 28', 'value': 0},
                        {'label': 'Mar 01 - Mar 21', 'value': 1},
                        {'label': 'Mar 22 - Mar 31', 'value': 2},
                        {'label': 'Apr 01 - May 04', 'value': 3}
                    ],
                    value=0,
                    id='tabs2'
                ),
        ], style={
            'width': '80%',
            'fontFamily': 'Sans-Serif',
            'height': 60,
            'margin-left': 'auto',
            'margin-right': 'auto'
        }),
        html.Div([
        dcc.Markdown(children=markdown_text[6]),
        dcc.Slider(
            id='maxdf-slider2',
            marks={i: '{}'.format(10.0**(i-4)) for i in range(5)},
            min=0,
            max=4,
            value=4,
            updatemode='drag',
            #vertical = True
        )], style={
            'width': '80%',
            'height': 60,
            'margin-left': 'auto',
            'margin-right': 'auto'
        }),
        html.Div([
        visdcc.Network(id='net',
                       options=dict(height='550px', width='100%')),
        ]),
    ], style={'marginLeft': '5%', 'marginRight': '5%'}),

])


@app.callback(Output('stock_price_and_tweet_mention', 'figure'), [Input('dropdown1', 'value')])
def stock_price_and_tweet_mention_count(symbol):
    # retrieve stock price data
    df1 = pd.read_csv('data/' +
                     'stock_price_change_' + symbol + '.csv')
    df1['DATE'] = pd.to_datetime(df1['DATE'], format='%d/%m/%y %H:%M')
    df1 = df1.set_index('DATE', verify_integrity=True)

    # retrieve tweet count
    keyword_dict = {'SIV': 'soybean', 'NATIONALUM': 'Aluminium', '399231': 'agriculture', 'AMZN': 'amazon',
                    'AAPL': 'apple', '002502': 'huawei', 'QCOM': 'qualcomm', 'VZ': 'verizon',
                    'C29': 'chip', 'NFLX': 'netflix', 'FB': 'facebook', 'BA': 'boeing', 'USDCNY': 'dollar',
                    'CNYUSD': 'yuan', 'ONGC': 'oil', '.IXIC': 'tech', 'T': 'at&t', 'INTC': 'intel', 'USDJPY': 'dollar',
                    'AVGO': 'broadcom', 'ACIA': 'acacia', 'GOOGL': 'google', 'IBM': 'ibm', 'TSLA': 'tesla', 'SMI': 'semiconductor',
                    'STLD': 'steel', 'WHGLY': 'pork', 'SIRC': 'solar'}

    keyword = keyword_dict[symbol]

    df2 = pd.read_csv('data/' +
                      'tweet_mention_count_' + keyword + '.csv')

    df2['Date'] = pd.to_datetime(df2['Date'], format='%Y-%m-%d %H:%M')
    df2 = df2.set_index('Date', verify_integrity=True)

    # construct the subplot structure
    trace1 = go.Scatter(
        x=df1.index,
        y=df1.diff().fillna(value=0).CLOSE, #df1.CLOSE-df1.OPEN #df1.diff().fillna(value=0).CLOSE # df1.CLOSE
        name='Stock return'
    )
    trace2 = go.Scatter(
        x=df2.index,
        y=df2.Count,
        name='Hourly tweet count'
    )
    fig = tools.make_subplots(rows=2, cols=1, specs=[[{}], [{}]],
                              shared_xaxes=True, shared_yaxes=True,
                              vertical_spacing=0.1)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 2, 1)

    return fig


@app.callback(
    Output('net', 'data'),
    [Input('tabs2', 'value'),
     Input('maxdf-slider2', 'value')])
def update_graph(tabs_value, maxdf_value):
    df_new = pd.read_csv(
        'data/' +
        'LDA_topics_' + str(transform_date(tabs_value)) + '_' + str(transform_maxdf(maxdf_value)) + '.csv')

    data = {'nodes': [{'id': 61, 'label': 'Trade War', 'color': '#fb91a3'}],
            'edges': []
           }

    count = 0
    clr = ['#fff060', '#caf8da', '#e0edff']
    for _, row in df_new.iterrows():
        topic_idx = int(row[0])
        topic_content = row[1]
        if topic_idx == 0:
            count += 1
        data['nodes'].append({'id': 20*(count-1) + topic_idx+1, 'label': topic_content, 'color': clr[count]})
        data['edges'].append({'id': '61-' + str(20*(count-1) + topic_idx+1), 'from': 61, 'to': 20*(count-1) + topic_idx+1},)

    return data


@app.callback(
    Output('wordcloud', 'src'),
    [Input('tabs1', 'value'),
     Input('maxdf-slider1', 'value')])
def update_wordcloud(tabs_value, maxdf_value):

    wordcloud_filename = 'data/' + str(transform_date(tabs_value)) + '/figure_tweet_wordCloud_' + str(transform_maxdf(maxdf_value)) + '_1000.png'
    wordcloud_encoded_image = base64.b64encode(open(wordcloud_filename, 'rb').read())
    src = 'data:image/png;base64,{}'.format(wordcloud_encoded_image.decode())
    return src


if __name__ == '__main__':
    app.run_server(debug=True)

