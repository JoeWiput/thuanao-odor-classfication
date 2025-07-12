import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import operator
import random


from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from scipy.spatial import distance
import xgboost as xgb 

from sklearn.metrics import accuracy_score,precision_score,recall_score,precision_recall_fscore_support, classification_report

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import argparse
import joblib


filename = './FS-taunao-data.xlsx'
data = pd.read_excel(filename, sheet_name='data')
thres_data = pd.read_excel(filename, sheet_name='thres')
odor_class={item:idx for idx,item in enumerate(set(thres_data["class"].to_list()))}

for idx, row in thres_data.iterrows():
    normalized_value = row['odor threshold (ug/kg)']
    data.loc[:, row['RI']] /= normalized_value 

label_mapping={'Savory(High sweet)': 0, 'Savory(Low sweet)': 1, 'Traditional(High sweet)': 2,'Traditional(Low sweet)':3}

reverse_label_mapping={v:k for k,v in label_mapping.items()}
X = data.drop(['class'],axis=1).to_numpy()
Y=np.zeros(X.shape[0])


for idx,label in enumerate(data['class'].to_numpy()):
    Y[idx]=label_mapping[label]

X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0, stratify=Y
    )

# - Training: 0.75 * 80% = 60%
# - Validation: 0.25 * 80% = 20%
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_val, Y_train_val, test_size=0.25, random_state=71, stratify=Y_train_val
    )

model_logit = LogisticRegression(random_state=77, solver='liblinear',max_iter=100)
model_svc = SVC(kernel='linear',probability=True)
model_knn = KNeighborsClassifier(n_neighbors=5,n_jobs=10,p=1)
model_xg = xgb.XGBClassifier()
model_dt=DecisionTreeClassifier(min_samples_leaf=15,criterion='entropy')

run_logit=True
run_svc=True
run_knn=True
run_xg=True
run_dt=True

models={'Logistic Regression':(model_logit,run_logit), 'SVM': (model_svc,run_svc),'KNN': (model_knn,run_knn),'XGBOOST':(model_xg,run_xg),"DCT":(model_dt ,run_dt)}

for name, (model,run) in models.items():
    if not run:
        continue

    model.fit(X_train,Y_train)
    pred_val=model.predict(X_val)

    # Compute metrics per class
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(Y_val, pred_val, average=None)

    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(Y_val, pred_val, average='macro')
    accuracy = accuracy_score(Y_val, pred_val)
    print(f'{name} acc: {accuracy} overall_precision: {overall_precision}  overall_recall: {overall_recall} overall_f1: {overall_f1}')
    model_file=f'{name}.pkl'
    print(f'{model_file} is saved')
    joblib.dump(model, model_file)

new_models=[]

for name, (model,run) in models.items():
    if not run:
        continue
    load_model_filename=f'{name}.pkl'
    m=joblib.load(load_model_filename)
    new_models.append((name,m))
    print(type(m))