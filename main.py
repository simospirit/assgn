#!/usr/bin/python

import sys
import pickle

import matplotlib.pyplot
from sklearn import preprocessing, decomposition
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

sys.path.append("../tools/")

from tools.feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


def showData(data_set, first_feature, second_feature):
    data = featureFormat(data_set, [first_feature, second_feature])
    for point in data:
        x = point[0]
        y = point[1]
        matplotlib.pyplot.scatter(x, y)

    matplotlib.pyplot.xlabel(first_feature)
    matplotlib.pyplot.ylabel(second_feature)
    matplotlib.pyplot.show()


### # ============= TASK 1 : DATASET EXPLORATION ... ================
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features

fin_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                      'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                      'director_fees']

e_features = ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                  'shared_receipt_with_poi']

POI_label = ['poi']

frac_features = ['fraction_to_this_person_from_poi', 'fraction_form_this_person_to_poi']

all_features_list = POI_label + fin_features + e_features + frac_features
all_features_list.remove('email_address')

my_features = ['total_stock_value', 'restricted_stock', 'exercised_stock_options', 'salary', 'bonus', 'deferred_income',
               'long_term_incentive']

features_list = POI_label + my_features + frac_features

# features_list = all_features_list
# features_list.remove('email_address')

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### # ========= TASK 2 : FUNCTIONS FOR REMOVING OUTLIERS ===========
# remove persons who don't have any feature value
for name in data_dict:
    data_without_feature = True
    for feature in data_dict.get(name):
        if (feature != 'poi' and data_dict.get(name)[feature] != 'NaN'):
            data_without_feature = False
            break
    if (data_without_feature == True):
        print ('Outliers name = ', name)

# Visualize data to identify outliers
showData(data_dict, 'total_stock_value', 'total_payments')
showData(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi')
showData(data_dict, 'salary', 'bonus')
showData(data_dict, 'restricted_stock', 'exercised_stock_options')
showData(data_dict, 'long_term_incentive', 'deferred_income')

data_dict.pop("TOTAL", 0)
data_dict.pop("LOCKHART EUGENE E", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

# Visualize data after removing outliers
showData(data_dict, 'total_stock_value', 'total_payments')
showData(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi')
showData(data_dict, 'salary', 'bonus')
showData(data_dict, 'restricted_stock', 'exercised_stock_options')
showData(data_dict, 'long_term_incentive', 'deferred_income')

### # ========= TASK 3: CREATE NEW FEATURES(S) ===========
# check how many people have email data
contain_email_data = 0
for name in data_dict:
    if (
            data_dict.get(name)['to_messages'] != 'NaN' and
            data_dict.get(name)['from_poi_to_this_person'] != 'NaN' and
            data_dict.get(name)['from_messages'] != 'NaN' and
            data_dict.get(name)['from_this_person_to_poi'] != 'NaN' and
            data_dict.get(name)['shared_receipt_with_poi'] != 'NaN' and
            data_dict.get(name)['email_address'] != 'NaN'

    ):
        contain_email_data = contain_email_data + 1

print ('Percentage of persons who has email data = ', (float(contain_email_data) / float(len(data_dict))) * 100, '%')

# create new features fraction_to_this_person_from_poi and fraction_form_this_person_to_poi
# since everyone has to have email data, just data is not available to us, I will put 0.3 as default value for persons
# who don't have email data.
for name in data_dict:
    if (data_dict.get(name)['to_messages'] != 'NaN' and data_dict.get(name)['from_poi_to_this_person'] != 'NaN' and
            data_dict.get(name)['from_messages'] != 'NaN' and data_dict.get(name)['from_this_person_to_poi'] != 'NaN'):
        data_dict.get(name)['fraction_to_this_person_from_poi'] = float(
            data_dict.get(name)['from_poi_to_this_person']) / float(data_dict.get(name)['from_messages'])
        data_dict.get(name)['fraction_form_this_person_to_poi'] = float(
            data_dict.get(name)['from_this_person_to_poi']) / float(data_dict.get(name)['to_messages'])
    else:
        data_dict.get(name)['fraction_to_this_person_from_poi'] = 0.3
        data_dict.get(name)['fraction_form_this_person_to_poi'] = 0.3

# count NaN values for every feature to see what feature are most reliable
feature_analysis = dict()
for feature in all_features_list:
    for name in data_dict:
        if (feature_analysis.get(feature, None) == None):
            feature_analysis[feature] = 0
        if (data_dict.get(name)[feature] == 'NaN'):
            feature_analysis[feature] = feature_analysis.get(feature) + 1

# checkout percentage of existing features data (loan_advances  has only 2.7397260274 % of data)
feature_analysis = sorted(feature_analysis.items(), key=lambda x: x[1], reverse=False)
for feature in feature_analysis:
    print (feature[0], ' = ', ((float(len(data_dict)) - float(feature[1])) / float(len(data_dict))) * 100, '%')

# fill missing feature values
for name in data_dict:
    for feature in data_dict[name]:
        if data_dict[name][feature] == 'NaN':
            data_dict[name][feature] = 0

### Store to my_dataset for easy export below.


my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, all_features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


clf = Pipeline([
    ('scaling', preprocessing.MinMaxScaler()),
    ('pca', decomposition.PCA(n_components = 15, svd_solver = 'full')),
    ('select_best', SelectKBest(k = 10)),
    ('algorithm', GaussianNB())
])

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project

from sklearn.model_selection  import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.



test_classifier(clf, my_dataset, all_features_list)

# print clf.named_steps['select_best'].scores_

dump_classifier_and_data(clf, my_dataset, all_features_list)
