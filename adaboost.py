"""
Author: Channy Hong
"""

import csv
import operator
import collections
import math
import random




#==============================================================================
#  Hyperparameters
#==============================================================================



T = 50
ranked_categories = ["num_concepts_mentioned", "prp_count", "Gerund_count", "NP_count", "VP_count", "count_pauses", "count_unintelligible", "count_trailing", "count_repetitions"]
numeric_categories = ["ttr", "R", "ARI", "CLI", "prp_noun_ratio", "word_sentence_ratio", "MLU", "SIM_score", "Bruten"]

num_train_per_label = 150
num_test_per_label = 50






#==============================================================================
#   Helper functions
#==============================================================================


def gini_numeric(category, data):
	candidates = []

	sorted_data = sorted(data, key=operator.itemgetter(category), reverse=False)
	y_label_counts = collections.Counter(datapoint['y_label'] for datapoint in sorted_data)

	num_less_positives = 0.0 # when category < midpoint && y_label == 1
	num_less_negatives = 0.0 # when category < midpoint && y_label == 0
	num_more_positives = float(y_label_counts[1]) # when category > midpoint && y_label == 1
	num_more_negatives = float(y_label_counts[0]) # when category > midpoint && y_label == 0

	for i in range(1, len(sorted_data)):

		midpoint = (sorted_data[i-1][category] + sorted_data[i][category]) / 2.0

		if sorted_data[i-1]['y_label'] == 0:
			num_less_negatives += 1.0
			num_more_negatives -= 1.0

		elif sorted_data[i-1]['y_label'] == 1:
			num_less_positives += 1.0
			num_more_positives -= 1.0

		num_less = num_less_negatives + num_less_positives
		num_more = num_more_negatives + num_more_positives

		less_gini = 1.0 - ((num_less_positives / num_less)**2) - ((num_less_negatives / num_less)**2)
		more_gini = 1.0 - ((num_more_positives / num_more)**2) - ((num_more_negatives / num_more)**2)

		num_all = num_less + num_more
		gini = ((num_less / num_all) * less_gini) + ((num_more / num_all) * more_gini)

		if (num_less_positives + num_more_negatives) >= (num_less_negatives + num_more_positives): 
			candidates.append({'threshold': midpoint, 'positive_direction': 'lesseq', 'gini': gini})
		elif (num_less_positives + num_more_negatives) < (num_less_negatives + num_more_positives):
			candidates.append({'threshold': midpoint, 'positive_direction': 'greater', 'gini': gini})

	# pick the the candidate with the lowest gini index
	winner = min(candidates, key=lambda x: x['gini'])

	return winner['threshold'], winner['positive_direction'], winner['gini']


def gini_ranked(category, data):
	candidates = []

	ranks = list(set([datapoint[category] for datapoint in data]))
	ranks = sorted(ranks)

	sorted_data = sorted(data, key=operator.itemgetter(category), reverse=False)
	y_label_counts = collections.Counter(datapoint['y_label'] for datapoint in sorted_data)

	num_lesseq_positives = 0.0 # when category < midpoint && y_label == 1
	num_lesseq_negatives = 0.0 # when category < midpoint && y_label == 0
	num_more_positives = float(y_label_counts[1]) # when category > midpoint && y_label == 1
	num_more_negatives = float(y_label_counts[0]) # when category > midpoint && y_label == 0

	current_rank_index = 0

	for i in range(len(sorted_data)):

		# base case: done with this current rank so calculate gini index and add to candidates, then move onto next rank
		if sorted_data[i][category] != ranks[current_rank_index]: 

			num_lesseq = num_lesseq_negatives + num_lesseq_positives
			num_more = num_more_negatives + num_more_positives

			lesseq_gini = 1.0 - ((num_lesseq_positives / num_lesseq)**2) - ((num_lesseq_negatives / num_lesseq)**2)
			more_gini = 1.0 - ((num_more_positives / num_more)**2) - ((num_more_negatives / num_more)**2)

			num_all = num_lesseq + num_more
			gini = ((num_lesseq / num_all) * lesseq_gini) + ((num_more / num_all) * more_gini)

			# add to candidates
			if (num_lesseq_positives + num_more_negatives) >= (num_lesseq_negatives + num_more_positives): 
				candidates.append({'threshold': ranks[current_rank_index], 'positive_direction': 'lesseq', 'gini': gini})
			elif (num_lesseq_positives + num_more_negatives) < (num_lesseq_negatives + num_more_positives):
				candidates.append({'threshold': ranks[current_rank_index], 'positive_direction': 'greater', 'gini': gini})

			# move on to next rank
			current_rank_index += 1

			# update memoized numbers
			if sorted_data[i]['y_label'] == 0:
				num_lesseq_negatives += 1.0
				num_more_negatives -= 1.0

			elif sorted_data[i]['y_label'] == 1:
				num_lesseq_positives += 1.0
				num_more_positives -= 1.0

		# main case: same rank, so keep updating memoized numbers
		else:
			if sorted_data[i]['y_label'] == 0:
				num_lesseq_negatives += 1.0
				num_more_negatives -= 1.0

			elif sorted_data[i]['y_label'] == 1:
				num_lesseq_positives += 1.0
				num_more_positives -= 1.0

	# pick the the candidate with the lowest gini index
	winner = min(candidates, key=lambda x: x['gini'])

	return winner['threshold'], winner['positive_direction'], winner['gini']





#==============================================================================
#   The adaptive boosting algorithm 
#==============================================================================


# Open pre-processed csv file and store as data (Python list)
fp = open("feature_set_dem.csv")
reader = csv.DictReader(fp)

positive_data = []
negative_data = []

all_categories = reader.fieldnames[:]
all_categories.remove("Category")

for csv_row in reader:
	datapoint = {}
	datapoint["weight"] = None
	datapoint["y_label"] = int(csv_row["Category"])

	for category in all_categories:
		if category in numeric_categories:
			datapoint[category] = float(csv_row[category])
		elif category in ranked_categories:
			datapoint[category] = int(csv_row[category])

	if datapoint["y_label"] == 0:
		negative_data.append(datapoint)

	elif datapoint["y_label"] == 1:
		positive_data.append(datapoint)

if (num_train_per_label + num_test_per_label > len(negative_data)) or (num_train_per_label + num_test_per_label > len(positive_data)):
	print "CHANGE the num data parameters!"

train_data = positive_data[:num_train_per_label] + negative_data[:num_train_per_label]
test_data = positive_data[num_train_per_label: num_train_per_label + num_test_per_label] + negative_data[num_train_per_label: num_train_per_label + num_test_per_label]


# initialize thresholds by identifying separation points via Gini index calculations
current_data = train_data[:]
random.shuffle(current_data)

# initial probability distribution of examples
num_datapoints = len(current_data)
initial_weight = 1.0/float(num_datapoints)

for datapoint in current_data:
	datapoint["weight"] = initial_weight


final_stumps = []

for _ in range(len(all_categories)):

	temporary_stumps = []

	for category in all_categories:

		category_data = [{category: datapoint[category], 'y_label': datapoint['y_label']} for datapoint in current_data]

		if category in numeric_categories:
			threshold, positive_direction, gini = gini_numeric(category, category_data)

		elif category in ranked_categories:
			threshold, positive_direction, gini = gini_ranked(category, category_data)

		# threshold: floating value
		# positive_direction: either 'lesseq' or 'greater' (for the 1 y_label classification)
		# gini: floating value -> ___
		# weight: floating value -> ____

		temporary_stumps.append({"category": category, "threshold": threshold, "positive_direction": positive_direction, "gini": gini, "weight": None})

	# reorder stumps in order of the gini scores and pick the stump with the lowest gini index
	sorted_temporary_stumps = sorted(temporary_stumps, key=operator.itemgetter("gini"), reverse=False)
	stump = sorted_temporary_stumps[0]

	# then delete the chosen stump from all_categories 
	all_categories.remove(stump["category"])

	# CALCULATE error, alpha

	error = None
	num_correct = 0
	num_incorrect = 0

	if stump["positive_direction"] == "lesseq":
		for datapoint in current_data:
			if datapoint[stump["category"]] <= stump["threshold"]:
				num_correct += 1 # correct case
			else:
				num_incorrect += 1 # incorrect case

	if stump["positive_direction"] == "greater":
		for datapoint in current_data:
			if datapoint[stump["category"]] > stump["threshold"]:
				num_correct += 1 # correct case
			else:
				num_incorrect += 1 # incorrect case

	error = float(num_incorrect) / (float(num_correct) + float(num_incorrect))

	if error < 0.01:
		error = 0.01
	elif error > 0.99:
		error = 0.99

	alpha = (1.0/2.0) * (math.log((1.0-error) / (error)))

	stump["weight"] = alpha
	final_stumps.append(stump)

	# UPDATE current data with new probabilities
	if stump["positive_direction"] == "lesseq":
		for datapoint in current_data:
			if datapoint[stump["category"]] <= stump["threshold"]:
				datapoint["weight"] = datapoint["weight"] * math.exp(-alpha) # correct case
			else:
				datapoint["weight"] = datapoint["weight"] * math.exp(alpha) # incorrect case

	if stump["positive_direction"] == "greater":
		for datapoint in current_data:
			if datapoint[stump["category"]] > stump["threshold"]:
				datapoint["weight"] = datapoint["weight"] * math.exp(-alpha) # correct case
			else:
				datapoint["weight"] = datapoint["weight"] * math.exp(alpha) # incorrect case

	# Normalize
	total_weight = 0.0
	for datapoint in current_data:
		total_weight += datapoint["weight"]
	
	factor = 1.0/total_weight

	for datapoint in current_data:
		datapoint["weight"] = factor * datapoint["weight"]

	# MAKE new dataset
	new_dataset = []
	for i in range(len(current_data)):
		# pick random number between 0 and 1
		random_num = random.uniform(0,1)

		# add up the weights until cumul_weight exceeds...
		counter = 0
		cumul_weight = current_data[0]["weight"]
		while (cumul_weight <= random_num):
			cumul_weight += current_data[counter+1]["weight"]
			counter += 1
			#print counter, cumul_weight, random_num

		# add this current data
		new_dataset.append(current_data[counter].copy())

	# UPDATE current_data with the new dataset
	current_data = new_dataset[:]

	# reset the weight of the examples
	num_datapoints = len(current_data)
	initial_weight = 1.0/float(num_datapoints)

	for datapoint in current_data:
		datapoint["weight"] = initial_weight








#==============================================================================
#   EVALUATION: with the resulting decision tree!!!
#==============================================================================



random.shuffle(test_data)

eval_num_correct = 0.0
eval_num_incorrect = 0.0

for test_datapoint in test_data:

	cumul_positive = 0.0
	cumul_negative = 0.0
	guess = None

	for stump in final_stumps:
		if stump["positive_direction"] == "lesseq":
			if test_datapoint[stump["category"]] <= stump["threshold"]:
				cumul_positive += stump["weight"] # correct case
			else:
				cumul_negative += stump["weight"] # incorrect case

		if stump["positive_direction"] == "greater":
			if test_datapoint[stump["category"]] > stump["threshold"]:
				cumul_positive += stump["weight"] # correct case
			else:
				cumul_negative += stump["weight"] # incorrect case

	if cumul_positive > cumul_negative:
		guess = 1
	else:
		guess = 0

	if guess is test_datapoint["y_label"]:
		eval_num_correct += 1.0
	else:
		eval_num_incorrect += 1.0


eval_accuracy = float(eval_num_correct) / (float(eval_num_correct) + float(eval_num_incorrect))

print eval_accuracy
print eval_num_correct, eval_num_incorrect









