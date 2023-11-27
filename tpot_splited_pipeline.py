import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator, ZeroCount
from sklearn.metrics import accuracy_score


# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('data/train.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('Cover_Type', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['Cover_Type'], random_state=None)

print("Training set: {} samples".format(training_features.shape[0]))
# Average CV score on the training set was: 0.8867577426611698
exported_pipeline = make_pipeline(
    ZeroCount(),
    ZeroCount(),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_leaf=2, min_samples_split=14)),
    StackingEstimator(estimator=LogisticRegression(C=25.0, dual=False, penalty="l2", max_iter=100000)),
    ZeroCount(),
    Normalizer(norm="max"),
    MinMaxScaler(),
    KNeighborsClassifier(n_neighbors=1, p=1, weights="uniform")
    , verbose=True)

exported_pipeline.fit(training_features, training_target)

print("Printing accuracy on test set")

results = exported_pipeline.predict(testing_features)
print(accuracy_score(testing_target,results))

print("Training on all data")
#Train on all data
exported_pipeline = make_pipeline(
    ZeroCount(),
    ZeroCount(),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_leaf=2, min_samples_split=14)),
    StackingEstimator(estimator=LogisticRegression(C=25.0, dual=False, penalty="l2", max_iter=1000000)),
    ZeroCount(),
    Normalizer(norm="max"),
    MinMaxScaler(),
    KNeighborsClassifier(n_neighbors=1, p=1, weights="uniform")
    , verbose=True)

exported_pipeline.fit(features, tpot_data['Cover_Type'])

#Predict
print("Reading test data")
tpot_data = pd.read_csv('data/test-full.csv', sep=',', dtype=np.float64)
print("Predicting")
results = exported_pipeline.predict(tpot_data)
print(results)
print("Saving results")
#Save results
def generating_submission_csv(prediction, name=""):
    data = {'Id': [k+1 for k in range(len(prediction))], 'Cover_type': prediction}

    # Créez un DataFrame Pandas à partir du dictionnaire
    predictions_df = pd.DataFrame(data)

    # Enregistrez le DataFrame dans un fichier CSV
    predictions_df.to_csv('predictions_'+name+'.csv', index=False)

generating_submission_csv(results, "tpot_splited_pipeline")

