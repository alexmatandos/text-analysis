import pandas
import pickle
import string
from nltk.corpus import stopwords

#the program for someone else to get the results that matter
#with 'pickle' replace 'write binary' with 'read binary' and 'dump' with 'load'

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

with open("objects.pickle", "rb") as f:
	machine = pickle.load(f)
	count_vectorizer_transformer = pickle.load(f)
	lemmatizer = pickle.load(f)

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