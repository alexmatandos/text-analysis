import pandas
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics

def run_kfold(data, target, split_number, machine, use_accuracy = 0, use_confusion = 0, use_2x2_confusion = 0):
	#print("run KFold")
	kfold_object = KFold(n_splits = split_number)
	kfold_object.get_n_splits(data)

	#creating matrices for the results
	results_r2 = []
	results_accuracy = []
	results_confusion = []
	results_2x2_confusion = []

	for training_index, test_index in kfold_object.split(data):
		data_training = data[training_index]
		target_training = target[training_index]
		data_test = data[test_index]
		target_test = target[test_index]
		#machine = linear_model.LinearRegression()
		machine.fit(data_training, target_training)
		#print(machine.score(data_training, target_training))
		new_target = machine.predict(data_test)
		results_r2.append(metrics.r2_score(target_test, new_target))

		if use_accuracy == 1:
			results_accuracy.append(metrics.accuracy_score(target_test, new_target))
		if use_confusion == 1:
			results_confusion.append(metrics.confusion_matrix(target_test, new_target))
		if use_2x2_confusion == 1:
			results_2x2_confusion.append(metrics.multilabel_confusion_matrix(target_test, new_target))
	
	return results_r2, results_accuracy, results_confusion, results_2x2_confusion

#The below establishes a condition for running the code only if certain criteria is met. That is, the variables used are the ones specified below.
if __name__ == '__main__':
	dataset = pandas.read_csv("dataset.csv")
	target = dataset.iloc[:, 0].values
	data = dataset.iloc[:, 3:9].values
	machine = linear_model.LinearRegression()
	run_kfold(data, target, machine, 4, 1, 1)


