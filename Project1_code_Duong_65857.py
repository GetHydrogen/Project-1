#%% import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib

#%% Part 1: Data Processing
df = pd.read_csv("Project 1 Data.csv")
print('Part 1: The Given Dataframe is: \n{}'.format(df))

#%% Part 2: Data Visualization
dfa = df.to_numpy()

scatter_X = [point[0] for point in dfa]
scatter_Y = [point[1] for point in dfa]
scatter_Z = [point[2] for point in dfa]
scatter_Step = [point[3] for point in dfa]

plt.scatter(scatter_Step,scatter_X, label='X',marker='o',s=10)
plt.scatter(scatter_Step,scatter_Y, label='Y',marker='s',s=10)
plt.scatter(scatter_Step,scatter_Z, label='Z',marker='D',s=10)

plt.title('Part 2: Scatter Plot of X,Y,Z based on Steps')
plt.xlabel('Maintenance Steps')
plt.ylabel('Coordinates')
plt.legend()
plt.savefig('Part 2 Scatter plot',dpi=200)
plt.show()

summary_stats = df.describe().drop(columns=['Step'])
print('Part 2: Statistical analysis of the dataset: \n{}'.format(summary_stats))

#%% Part 3: Correlation Analysis
corr_matrix = df.corr()
corr_target = corr_matrix['Step']
corr_features = corr_matrix.index[0:3]

plt.figure()
sns.barplot(x=corr_features, y=corr_target[corr_features])
plt.title('Part 3: X,Y,Z Correlations with Maintenance Steps')
plt.savefig('Part 3 Correlation plot',dpi=200)
plt.show()

print('\nPart 3: Correlation analysis of the dataset: \n{}'.format(corr_matrix))

#%% Part 4: Classification Model Development/Engineering
#   4.1 Stratified Random Sampling
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df["Step"]):
    strat_train_set = df.loc[train_index].reset_index(drop=True)
    strat_test_set = df.loc[test_index].reset_index(drop=True)

train_y = strat_train_set['Step']
test_y = strat_test_set['Step']
strat_train_set = strat_train_set.drop(columns=["Step"], axis = 1)
strat_test_set = strat_test_set.drop(columns=["Step"], axis = 1)

#%% 4.2 Feature Scaling
from sklearn.preprocessing import MinMaxScaler
my_scaler = MinMaxScaler()
scaled_data = my_scaler.fit_transform(strat_train_set)
scaled_data_df = pd.DataFrame(scaled_data)
train_X = strat_train_set
test_X = strat_test_set

#%% 4.3 Classfication models creation - GridSearch CV
from sklearn.model_selection import GridSearchCV

#%% 4.3.1 Support Vector Machine - Classifier
from sklearn.svm import SVC
model1 = SVC()
param_grid1 = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf','poly'],
    'gamma': [0.001, 0.01, 0.1, 'scale', 'auto'],
}
grid_search1 = GridSearchCV(model1, param_grid1, cv=5, scoring='accuracy')
grid_search1.fit(train_X, train_y)
best_hyperparams1 = grid_search1.best_params_
best_model1 = grid_search1.best_estimator_
y_pred1 = best_model1.predict(test_X) #Use test feature to predict target by SVC

#%% 4.3.2 K-Nearest Neighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
param_grid2 = {'n_neighbors': [3, 4, 5, 6, 7, 8, 9],}
model2 = KNeighborsClassifier()
grid_search2 = GridSearchCV(model2, param_grid2, cv=5, scoring='accuracy')
grid_search2.fit(train_X, train_y)
best_hyperparameters2 = grid_search2.best_params_
best_model2 = grid_search2.best_estimator_
y_pred2 = best_model2.predict(test_X) #Use test feature to predict target by KNN

#%% 4.3.3 Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
param_grid3 = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
model3 = RandomForestClassifier()
grid_search3 = GridSearchCV(model3, param_grid3, cv=5, scoring='accuracy')
grid_search3.fit(train_X, train_y)
best_hyperparameters3 = grid_search3.best_params_
best_model3 = grid_search3.best_estimator_
y_pred3 = best_model3.predict(test_X) #Use test feature to predict target by RFC

#%% Part 5: Model Performance Analysis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

#The Following matrix shows the F1 Score, Precision and Accuracy of the 3 models
classifier_perf = {
    'SVC': [f1_score(test_y, y_pred1, average='macro'), precision_score(test_y, y_pred1, average='macro'), accuracy_score(test_y, y_pred1)],
    'K-NN': [f1_score(test_y, y_pred2, average='macro'), precision_score(test_y, y_pred2, average='macro'), accuracy_score(test_y, y_pred2)],
    'RFC': [f1_score(test_y, y_pred3, average='macro'), precision_score(test_y, y_pred3, average='macro'), accuracy_score(test_y, y_pred3)]
}
classifier_perf = pd.DataFrame(classifier_perf,['F1 Score','Precision','Accuracy'])
print('\nThe performance of the 3 Machine Learning models: \n{}'.format(classifier_perf))

print('=>Support Vector Classifier has the highest F1 Score, Precision and Accuracy\n')

cm1 = confusion_matrix(test_y, y_pred3)
plt.figure()
sns.heatmap(cm1)
plt.xlabel('Predicted Step from SVM')
plt.ylabel('Test Set')
plt.title('Part 5: Confusion Matrix of Random Forest Classifier')
plt.savefig('Part 5 Confusion Matrix',dpi=200)
plt.show()

#%% Part 6: Model Evaluation
joblib.dump(grid_search1, 'Duong_SVM.joblib')
loaded_model = joblib.load('Duong_SVM.joblib')

p6_data = [[9.375,3.0625,1.51],
        [6.995,5.125,0.3875],
        [0,3.0625,1.93],
        [9.4,3,1.8],
        [9.4,3,1.3]]
p6_df = pd.DataFrame(p6_data, columns=['X', 'Y', 'Z'])

p6_pred = loaded_model.predict(p6_df)
p6_pred = pd.DataFrame(p6_pred, columns=['Step'])
p6_join = p6_df.join(p6_pred)

print('Part 6: The SVC model prediction for provided data set is: \n{}'.format(p6_join))

plt.scatter(df['Step'],df['X'], label='X_p1',marker='x',s=10)
plt.scatter(df['Step'],df['Y'], label='X_p1',marker='+',s=20)
plt.scatter(df['Step'],df['Z'], label='X_p1',marker='.',s=10)

plt.scatter(p6_join['Step'],p6_join['X'], label='X_p6',marker='o',s=60)
plt.scatter(p6_join['Step'],p6_join['Y'], label='Y_p6',marker='s',s=60)
plt.scatter(p6_join['Step'],p6_join['Z'], label='Z_p6',marker='D',s=60)

plt.title('Part 6: Comparison of Predicted to Original Data')
plt.xlabel('Maintenance Steps')
plt.ylabel('Coordinates')
plt.legend()
plt.savefig('Part 6 Scatter compare',dpi=200)
plt.show()