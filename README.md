# TradeWar_on_Twitter

Years of trade tensions between the world’s two biggest economies have taken a worse turn since the start of 2018. President Donald Trump's "trade war" officially began on March 1 when he announced steep tariffs on steel and aluminium, which were then signed into effect on March 8. The situation escalated since March 22, when U.S. Trade Representative proposed 25 percent duties on Chinese products and China quickly fought back by levying tariffs on $3 billion of U.S. imports. From April 03 onwards, the focus is shifted to Intellectual Property and high-tech industrial products.	

I am proposing a project to gain more insights on the trade war through big data analysis with tweets. I propose two directions to work on for this project. 1) News events related to "trade war" demonstrated great impact on stock market volatility. Market movement is a result of population behavior. Market indices plunged in response to news reflects population's pessimistic opinions over the outcome of a trade war. 2) As the tensions escalate over time, more people have come to realize that "trade war" has gone beyond a politic/economic topic, but will closely affect our daily life in the near future. By analyzing the topics that are co-mentioned in tweets with "trade war", I wish to identify the insightful predictions made by the general public that are likely to become true, and provide us a better picture of when and how is the trade war going to affect our life. For example, is there any field of industry that we should avoid in future job hunting, as it may be closely affected by the outcome of the trade war? As the old saying goes, “two heads are better than one”. In the era of big data, we should hear as many voices as possible before reaching a conclusion.


Project is under development. 

Update 2018/05/01

View XiaoyuZhang_TradeWar_twitterAnalysis.ipynb for results and analyses. 

*.csv files contain processed tweets to be analyzed by XiaoyuZhang_TradeWar_twitterAnalysis.ipynb 

./figures/ contains the saved figures for this project. 


Update 2018/05/17

Included the codes for retrieving historical tweets and collecting streaming tweets in respective folders. 

Included the codes for deploying the results dashboard onto Heroku. 

Please visit https://tradewar-on-twitter.herokuapp.com for the most updated results. 


Project Reflections:

In retrospect, one may be surprised to find out that, sometimes people are able to predict the future correctly. It is based on their backgrounds, experiences, concerns, and the information that they happen to have collected correctly regarding the problem. Just like one tree in a trained random forest model that happens to utilize the correct subset of data available and make the correct decision, I am interested in finding out whether there are such tweets that we can learn from before the future arrives.

Having that in mind, the main idea of this project is not to summarize the mainstream ideas that occupy 90% of the data, but to find efficient ways to identify less popular but insightful topics. The assumption that I made is that, these topics are usually associated lower document frequency than the mainstream topics but slightly higher than completely rubbish ideas, for they are usually recognized/re-iterated at the level of a minor group of people. 
