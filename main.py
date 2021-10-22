from folktables import ACSDataSource, BasicProblem, adult_filter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pickle
from sklearn import svm
import numpy as np

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
acs_data = data_source.get_data(download=False)#download=True
#print(acs_data)
print(2)
ACSIncomeNew = BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
    ],
    target='PINCP',
    target_transform=lambda x: x > 50000,
    group='RAC1P',
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)
print(ACSIncomeNew)

X, y, _ = ACSIncomeNew.df_to_numpy(acs_data)
print(4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Plug-in your method for tabular datasets
#model = LogisticRegression(max_iter=1000)
model = AdaBoostClassifier(n_estimators=200, random_state=6)
   
# adaboost: 0.7943977170321418
# [[180709  29665]
#  [ 38780  83746]]

# Train on CA data
model.fit(X_train, y_train)

# Test on MI data
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))
cm = confusion_matrix(y_test, y_pred)
print(cm)
tp=cm[0][0]
fp = cm[0][1]
fn=cm[1][0]
tn=cm[1][1]

fpr = fp / (fp + tn)
tpr = tp / (tp + fn)
print(fpr, tpr)
#
# #50K:
# #acc 0.7539561429858816
# #fpr 0.30832247956660436 tpr 0.783032086928168
#
#
# #10K
# #0.9000210273355362
# #0.08745713367735675 0.7630370848576142
# #20K
# #0.825866626614599
# #0.1586939397413471 0.7590681450903215
# #50K:
# #acc 0.7539561429858816
# #fpr 0.30832247956660436 tpr 0.783032086928168
# #75K
# #0.8240042054671073
# #0.3708816096886473 0.8521003320112458
# #100K
# #0.8894863322319014
# #0.3478847451093765 0.8971236159901774
