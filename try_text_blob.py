from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

tb_object = TextBlob("Excellent! Easy to use. Fast delivery.", analyzer = NaiveBayesAnalyzer())
print(tb_object.sentiment)