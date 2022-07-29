import nltk
import pandas
import json
import kfold_template
import string
#nltk.download("stopwords")

#nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

#print(stopwords.words("english"))

#remember that all that info separated by commas between the brackets "{}", represent only one json file

review_text = []
review_star = []

with open('yelp_review_part.json', encoding = "utf-8") as f:
	for line in f:
		json_line = json.loads(line)
		review_star.append(json_line["stars"])
		review_text.append(json_line["text"])

dataset = pandas.DataFrame(data = {"text": review_text, "stars": review_star})
#print(dataset)
#print(dataset.shape)
#use only the first 3000 observations

dataset_cropped = dataset[0:3000]

#merging 2, 3, and 4 stars into one group
dataset_cropped['stars'] = dataset_cropped['stars'].replace(2, 3)
dataset_cropped['stars'] = dataset_cropped['stars'].replace(4, 3)

#dropping 2 and 4 stars observations, however remember to reset the index (since kfold split dataset through the index)
#dataset_cropped = dataset_cropped[dataset_cropped['stars'] == 1	| dataset_cropped['stars'] == 3 | dataset_cropped['stars'] == 5]
#dataset_cropped = dataset_cropped.reset_index(drop = True, inplace = True)

target = dataset_cropped['stars']
data = dataset_cropped['text']


lemmatizer = WordNetLemmatizer()

def pre_processing(text):
	text = text.translate(str.maketrans('', '', string.punctuation))
	#the above "translate" a string section by certain criteria
	text_processed = text.split()
	result = []
	for word in text_processed:
		lower_word = word.lower()
		result.append(lower_word)
		if lower_word not in stopwords.words("english"):
			lower_word = lemmatizer.lemmatize(lower_word)
			result.append(lower_word)
	return result

print(pre_processing("Total bill for this horrible service? Over $8Gs."))

count_vectorizer_transformer = CountVectorizer(analyzer = pre_processing).fit(data)
data = count_vectorizer_transformer.transform(data)

machine = MultinomialNB()
results = kfold_template.run_kfold(data, target, 4, machine, 1, 1, 1)

print(results[1])

for result in results[2]:
	print(result)

#when predicting data, remember to transform string variable with count_vectorizer_transformer.transform

machine = MultinomialNB()
machine.fit(data, target)

#excerpt = "Total bill for this horrible service? Over $8Gs."
#excerpt = count_vectorizer_transformer.transform([excerpt])
#prediction = machine.predict(excerpt)
#prediction_probability = machine.predict_proba(excerpt)
#print(prediction)
#print(prediction_probability)

#predicting for multiple reviews

new_reviews = pandas.read_csv("new_reviews.csv", header = None)
excerpts = count_vectorizer_transformer.transform(new_reviews.iloc[:, 0])
prediction = machine.predict(excerpts)
prediction_probability = machine.predict_proba(excerpts)
print(prediction)
print(prediction_probability)

new_reviews['prediction'] = prediction
prediction_probability_dataframe = pandas.DataFrame(prediction_probability)
new_reviews = pandas.concat([new_reviews, prediction_probability_dataframe], axis = 1)

new_reviews = new_reviews.rename(columns = {new_reviews.columns[0]: "text", new_reviews.columns[1]: "prediction", new_reviews.columns[2]: "prediction_probability_1", new_reviews.columns[3]: "prediction_probability_3", new_reviews.columns[4]: "prediction_probability_5"})

new_reviews['prediction'] = new_reviews['prediction'].astype(int)
new_reviews['prediction_probability_1'] = round(new_reviews['prediction_probability_1'], 5)
new_reviews['prediction_probability_3'] = round(new_reviews['prediction_probability_3'], 5)
new_reviews['prediction_probability_5'] = round(new_reviews['prediction_probability_5'], 5)

new_reviews.to_csv("new_reviews_with_prediction.csv", index = False, float_format = '%.9f')