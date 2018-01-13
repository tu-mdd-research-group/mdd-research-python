# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:24:32 2018

@author: youpeng
"""

# In[]:

import pandas as pd
camb_fltr = pd.read_csv("camb_fltr.expr.csv")
jap_fltr = pd.read_csv("jap_fltr.expr.csv")
libr_fltr = pd.read_csv("libr_fltr.expr.csv")

camb_class = pd.read_csv("camb_class.csv")
jap_class = pd.read_csv("jap_class.csv")
libr_class = pd.read_csv("libr_class.csv")

camb = pd.concat([camb_fltr, camb_class], axis=1)
jap = pd.concat([jap_fltr, jap_class], axis=1)
libr = pd.concat([libr_fltr, libr_class], axis=1)

combine =  pd.concat([camb, jap, libr])
combine = combine.sort_values('class')
combine = combine.reset_index(drop=True)
combine.to_pickle('all_data_from_three_sources')

import matplotlib.pyplot as plt

# In[]:

'''
this session focus on people variance
'''

X = combine.iloc[:,1:-1]
y = combine.iloc[:,-1]
print('Number of genes:', X.shape[1])
plt.figure(figsize=(15, 3))
plt.title('First control, second MDD boxplot')
plt.boxplot(X)
plt.show()
plt.figure(figsize=(15, 3))
plt.title('First control, second MDD standard deviation')
plt.plot(X.T.std())
plt.show()
plt.figure(figsize=(15, 3))
plt.title('First control, second MDD mean')
plt.plot(X.T.mean())
plt.show()

# In[]:

# control only
X_control = combine.loc[combine.iloc[:,-1]==0].iloc[:,1:-1]
plt.figure(figsize=(15, 3))
plt.title('Control boxplot')
plt.boxplot(X_control)
plt.show()
plt.figure(figsize=(15, 3))
plt.title('Control standard deviation')
plt.plot(X_control.T.std())
plt.show()
plt.figure(figsize=(15, 3))
plt.title('Control mean')
plt.plot(X_control.T.mean())
plt.show()

# In[]:

# MDD only
X_mdd = combine.loc[combine.iloc[:,-1]==1].iloc[:,1:-1]
plt.figure(figsize=(15, 3))
plt.title('MDD boxplot')
plt.boxplot(X_mdd)
plt.show()
plt.figure(figsize=(15, 3))
plt.title('MDD standard deviation')
plt.plot(X_mdd.T.std())
plt.show()
plt.figure(figsize=(15, 3))
plt.title('MDD mean')
plt.plot(X_mdd.T.mean())
plt.show()

# In[]:

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from mpl_toolkits.mplot3d import Axes3D

X = combine.iloc[:,1:-1]
y = combine.iloc[:,-1]
target_names = ['health', 'MDD']
colors = ['navy', 'turquoise']
lw = 2

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
#plt.figure(figsize=(7,5))
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of MDD dataset before chi2 Pre-select')
plt.show()

fig = plt.figure()
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions before chi2 Pre-select")
ax.set_xlabel("1st D")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd D")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd D")
ax.w_zaxis.set_ticklabels([])
plt.show()


# In[]:

# use pre-select to help filter
ch2 = SelectKBest(chi2, k=13)
X = ch2.fit_transform(X, y)
pca = PCA(n_components=3)
X_r = pca.fit(X).transform(X)
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
#plt.figure(figsize=(7,5))
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of MDD dataset after chi2 Pre-select')
plt.show()

fig = plt.figure()
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions after chi2 Pre-select")
ax.set_xlabel("1st D")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd D")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd D")
ax.w_zaxis.set_ticklabels([])
plt.show()


# In[]:

'''
the second method selectkbest+pca shows better results
'''

import numpy as np

X = combine.iloc[:,1:-1]
y = combine.iloc[:,-1]
target_names = ['health', 'MDD']
colors = ['navy', 'turquoise']
lw = 2
k_list = np.linspace(4,100,20).astype(int)

for k_ in k_list:

    ch2 = SelectKBest(chi2, k=k_)
    X_s = ch2.fit_transform(X, y)
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X_s)
    print('k value is:', k_)
    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))
    fig = plt.figure(1)
    for color, i, target_name in zip(colors, [0, 1], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of MDD dataset after chi2 Pre-select')
    plt.show()
    
    fig = plt.figure(2)
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(X_s)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
               cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title("First three PCA directions after chi2 Pre-select")
    ax.set_xlabel("1st D")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd D")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd D")
    ax.w_zaxis.set_ticklabels([])

    plt.show()

# k bests are 44, 34, 39, 19, 14, 

# In[]:
from sklearn.model_selection import train_test_split

X_all = combine.iloc[:,1:-1]
y_all = combine.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.33, random_state=42, stratify = y_all)
X = X_train.reset_index(drop=True)
X_outside_test = X_test.reset_index(drop=True)
y = y_train.reset_index(drop=True)
y_outside_test = y_test.reset_index(drop=True)

'''
X_all.shape
Out[156]: (269, 5301)

X_train.shape
Out[157]: (172, 5301)

y_train.shape
Out[158]: (172,)

X_outside_test.shape
Out[159]: (89, 5301)

172/10
Out[160]: 17.2

# Stratified-10-fold-cv-test, each time train 90% and validate 10%
# Repeat Stratified-10-fold-cv-test 40 times process to merge the randomization.

'''

# In[]:

import time
start_time = time.time()

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score

# X = combine.iloc[:,1:-1]
# y = combine.iloc[:,-1]
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=40, random_state=34)
pipe = make_pipeline(PCA(n_components=10), SVC())
val_list = []
test_list = []

for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipe.fit(X_train, y_train)
    val_list.append(f1_score(y_test, pipe.predict(X_test)))
    test_list.append(f1_score(y_outside_test, pipe.predict(X_outside_test)))
    
plt.figure(figsize=(13, 5))
plt.plot(val_list, label='validate')
plt.plot(test_list, label='test')
plt.title('test and validate f1 score')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

val = pd.DataFrame(val_list).describe()
print('SVM CV result 1', val)
tst = pd.DataFrame(test_list).describe()
print('SVM test result 1', tst)
print("--- %s seconds ---" % (time.time() - start_time))

'''
SVM CV result 1                 0
count  400.000000
mean     0.788143
std      0.027037
min      0.692308
25%      0.774194
50%      0.785714
75%      0.814815
max      0.880000
SVM test result 1                 0
count  400.000000
mean     0.796871
std      0.007867
min      0.774648
25%      0.794118
50%      0.797101
75%      0.802920
max      0.817518
--- 21.384427309036255 seconds ---
'''

# In[]:

start_time = time.time()

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=40, random_state=34)
pipe = make_pipeline(PCA(n_components=10), SVC(class_weight='balanced'))
val_list = []
test_list=[]

for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipe.fit(X_train, y_train)
    val_list.append(f1_score(y_test, pipe.predict(X_test)))
    test_list.append(f1_score(y_outside_test, pipe.predict(X_outside_test)))
    
plt.figure(figsize=(13, 5))
plt.plot(val_list, label='validate')
plt.plot(test_list, label='test')
plt.title('test and validate f1 score')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

val = pd.DataFrame(val_list).describe()
print('SVM CV result 2', val)
tst = pd.DataFrame(test_list).describe()
print('SVM test result 2', tst)
print("--- %s seconds ---" % (time.time() - start_time))

'''
SVM CV result 2                 0
count  400.000000
mean     0.788280
std      0.044612
min      0.640000
25%      0.758621
50%      0.785714
75%      0.814815
max      0.916667
SVM test result 2                 0
count  400.000000
mean     0.791313
std      0.010483
min      0.763359
25%      0.783950
50%      0.791045
75%      0.800000
max      0.824427
--- 22.00201725959778 seconds ---
'''

#------------------------------------------------------------

# In[]:

from sklearn import linear_model 
from sklearn.pipeline import Pipeline

start_time = time.time()
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=40, random_state=34)
logistic = linear_model.LogisticRegression(C=1e5)
pca = PCA(n_components=10)
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
val_list = []
test_list=[]

for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipe.fit(X_train, y_train)
    val_list.append(f1_score(y_test, pipe.predict(X_test)))
    test_list.append(f1_score(y_outside_test, pipe.predict(X_outside_test)))
    
plt.figure(figsize=(13, 5))
plt.plot(val_list, label='validate')
plt.plot(test_list, label='test')
plt.title('test and validate f1 score')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

val = pd.DataFrame(val_list).describe()
print('Logistic Regression CV result 1', val)
tst = pd.DataFrame(test_list).describe()
print('Logistic Regression test result 1', tst)
print("--- %s seconds ---" % (time.time() - start_time))

'''
Logistic Regression CV result 1                 0
count  400.000000
mean     0.746542
std      0.068141
min      0.454545
25%      0.695652
50%      0.750000
75%      0.800000
max      0.909091
Logistic Regression test result 1                 0
count  400.000000
mean     0.770710
std      0.017343
min      0.715447
25%      0.759690
50%      0.771654
75%      0.782466
max      0.825397
--- 21.046109914779663 seconds ---
'''

# In[]:

start_time = time.time()
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=40, random_state=34)
logistic = linear_model.LogisticRegression(C=1e5, class_weight = 'balanced')
pca = PCA(n_components=10)
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
val_list = []
test_list=[]

for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipe.fit(X_train, y_train)
    val_list.append(f1_score(y_test, pipe.predict(X_test)))
    test_list.append(f1_score(y_outside_test, pipe.predict(X_outside_test)))
    
plt.figure(figsize=(13, 5))
plt.plot(val_list, label='validate')
plt.plot(test_list, label='test')
plt.title('test and validate f1 score')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

val = pd.DataFrame(val_list).describe()
print('Logistic Regression CV result 2', val)
tst = pd.DataFrame(test_list).describe()
print('Logistic Regression test result 2', tst)
print("--- %s seconds ---" % (time.time() - start_time))

'''
Logistic Regression CV result 2                 0
count  400.000000
mean     0.688003
std      0.108203
min      0.266667
25%      0.629934
50%      0.695652
75%      0.761905
max      0.952381
Logistic Regression test result 2                 0
count  400.000000
mean     0.720166
std      0.019709
min      0.666667
25%      0.704762
50%      0.720721
75%      0.733945
max      0.778761
--- 22.51801562309265 seconds ---
'''

#------------------------------------------------------------

# In[]:

from sklearn import tree 
from sklearn.pipeline import Pipeline

start_time = time.time()
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=40, random_state=34)
dectree = tree.DecisionTreeClassifier()
pca = PCA(n_components=10)
pipe = Pipeline(steps=[('pca', pca), ('tree', dectree)])
val_list = []
test_list=[]

for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipe.fit(X_train, y_train)
    val_list.append(f1_score(y_test, pipe.predict(X_test)))
    test_list.append(f1_score(y_outside_test, pipe.predict(X_outside_test)))
    
plt.figure(figsize=(13, 5))
plt.plot(val_list, label='validate')
plt.plot(test_list, label='test')
plt.title('test and validate f1 score')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

val = pd.DataFrame(val_list).describe()
print('Decision Tree Classifier CV result 1', val)
tst = pd.DataFrame(test_list).describe()
print('Decision Tree Classifier test result 1', tst)
print("--- %s seconds ---" % (time.time() - start_time))

'''
Decision Tree Classifier CV result 1                 0
count  400.000000
mean     0.679428
std      0.103816
min      0.210526
25%      0.608696
50%      0.695652
75%      0.750000
max      0.916667
Decision Tree Classifier test result 1                 0
count  400.000000
mean     0.739555
std      0.035724
min      0.619469
25%      0.717707
50%      0.741379
75%      0.766667
max      0.820513
--- 23.644083976745605 seconds ---
'''


# In[]:

start_time = time.time()
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=40, random_state=34)
dectree = tree.DecisionTreeClassifier(class_weight='balanced')
pca = PCA(n_components=10)
pipe = Pipeline(steps=[('pca', pca), ('tree', dectree)])
val_list = []
test_list=[]

for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipe.fit(X_train, y_train)
    val_list.append(f1_score(y_test, pipe.predict(X_test)))
    test_list.append(f1_score(y_outside_test, pipe.predict(X_outside_test)))
    
plt.figure(figsize=(13, 5))
plt.plot(val_list, label='validate')
plt.plot(test_list, label='test')
plt.title('test and validate f1 score')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

val = pd.DataFrame(val_list).describe()
print('Decision Tree Classifier CV result 2', val)
tst = pd.DataFrame(test_list).describe()
print('Decision Tree Classifier test result 2', tst)
print("--- %s seconds ---" % (time.time() - start_time))


'''
Decision Tree Classifier CV result 2                 0
count  400.000000
mean     0.683824
std      0.107577
min      0.333333
25%      0.615385
50%      0.695652
75%      0.761905
max      0.916667
Decision Tree Classifier test result 2                 0
count  400.000000
mean     0.730786
std      0.038093
min      0.588235
25%      0.706643
50%      0.735043
75%      0.757084
max      0.813559
--- 23.257717847824097 seconds ---
'''

#------------------------------------------------------------

# In[]:

from sklearn import ensemble 
from sklearn.pipeline import Pipeline

start_time = time.time()
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=40, random_state=34)
rftree = ensemble.RandomForestClassifier() # default is 10 trees
pca = PCA(n_components=10)
pipe = Pipeline(steps=[('pca', pca), ('tree', rftree)])
val_list = []
test_list=[]

for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipe.fit(X_train, y_train)
    val_list.append(f1_score(y_test, pipe.predict(X_test)))
    test_list.append(f1_score(y_outside_test, pipe.predict(X_outside_test)))
    
plt.figure(figsize=(13, 5))
plt.plot(val_list, label='validate')
plt.plot(test_list, label='test')
plt.title('test and validate f1 score')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

val = pd.DataFrame(val_list).describe()
print('Random Forest Classifier CV result 1 --- 10', val)
tst = pd.DataFrame(test_list).describe()
print('Random Forest Classifier test result 1 --- 10', tst)
print("--- %s seconds ---" % (time.time() - start_time))


'''
Random Forest Classifier CV result 1 --- 10                 0
count  400.000000
mean     0.694503
std      0.105956
min      0.333333
25%      0.636364
50%      0.720000
75%      0.769231
max      0.956522
Random Forest Classifier test result 1 --- 10                 0
count  400.000000
mean     0.759800
std      0.031359
min      0.653846
25%      0.739496
50%      0.760331
75%      0.779661
max      0.848000
--- 25.944265604019165 seconds ---
'''


# In[]:

start_time = time.time()
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=40, random_state=34)
rftree = ensemble.RandomForestClassifier(class_weight='balanced')
pca = PCA(n_components=10)
pipe = Pipeline(steps=[('pca', pca), ('tree', rftree)])
val_list = []
test_list=[]

for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipe.fit(X_train, y_train)
    val_list.append(f1_score(y_test, pipe.predict(X_test)))
    test_list.append(f1_score(y_outside_test, pipe.predict(X_outside_test)))
    
plt.figure(figsize=(13, 5))
plt.plot(val_list, label='validate')
plt.plot(test_list, label='test')
plt.title('test and validate f1 score')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

val = pd.DataFrame(val_list).describe()
print('Random Forest Classifier CV result 2 --- 10', val)
tst = pd.DataFrame(test_list).describe()
print('Random Forest Classifier test result 2 --- 10', tst)
print("--- %s seconds ---" % (time.time() - start_time))


'''
Random Forest Classifier CV result 2 --- 10                 0
count  400.000000
mean     0.692981
std      0.107103
min      0.333333
25%      0.627530
50%      0.695652
75%      0.769231
max      0.956522
Random Forest Classifier test result 2 --- 10                 0
count  400.000000
mean     0.753210
std      0.031243
min      0.660377
25%      0.735043
50%      0.752137
75%      0.774194
max      0.842975
--- 25.955793857574463 seconds ---
'''

#------------------------------------------------------------

# In[]:

from sklearn import ensemble 
from sklearn.pipeline import Pipeline

start_time = time.time()
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=40, random_state=34)
rftree = ensemble.RandomForestClassifier(n_estimators=500)
pca = PCA(n_components=10)
pipe = Pipeline(steps=[('pca', pca), ('tree', rftree)])
val_list = []
test_list=[]

for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipe.fit(X_train, y_train)
    val_list.append(f1_score(y_test, pipe.predict(X_test)))
    test_list.append(f1_score(y_outside_test, pipe.predict(X_outside_test)))
    
plt.figure(figsize=(13, 5))
plt.plot(val_list, label='validate')
plt.plot(test_list, label='test')
plt.title('test and validate f1 score')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

val = pd.DataFrame(val_list).describe()
print('Random Forest Classifier CV result 3', val)
tst = pd.DataFrame(test_list).describe()
print('Random Forest Classifier test result 3', tst)
print("--- %s seconds ---" % (time.time() - start_time))

'''
Random Forest Classifier CV result 3                 0
count  400.000000
mean     0.750782
std      0.076221
min      0.521739
25%      0.695652
50%      0.758621
75%      0.800000
max      0.960000
Random Forest Classifier test result 3                 0
count  400.000000
mean     0.795091
std      0.020307
min      0.736000
25%      0.781250
50%      0.795321
75%      0.809524
max      0.859504
--- 252.05404710769653 seconds ---
'''


#------------------------------------------------------------

# In[]:

from sklearn import ensemble 
from sklearn.pipeline import Pipeline

start_time = time.time()
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=40, random_state=34)
rftree = ensemble.RandomForestClassifier(n_estimators=1000)
pca = PCA(n_components=10)
pipe = Pipeline(steps=[('pca', pca), ('tree', rftree)])
val_list = []
test_list=[]

for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipe.fit(X_train, y_train)
    val_list.append(f1_score(y_test, pipe.predict(X_test)))
    test_list.append(f1_score(y_outside_test, pipe.predict(X_outside_test)))

plt.figure(figsize=(13, 5))
plt.plot(val_list, label='validate')
plt.plot(test_list, label='test')
plt.title('test and validate f1 score')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

val = pd.DataFrame(val_list).describe()
print('Random Forest Classifier CV result 4 --- 1000', val)
tst = pd.DataFrame(test_list).describe()
print('Random Forest Classifier test result 4 --- 1000', tst)
print("--- %s seconds ---" % (time.time() - start_time))

'''
Random Forest Classifier CV result 4 --- 1000                 0
count  400.000000
mean     0.752245
std      0.077244
min      0.500000
25%      0.695652
50%      0.750000
75%      0.800000
max      0.960000
Random Forest Classifier test result 4 --- 1000                 0
count  400.000000
mean     0.795408
std      0.019616
min      0.742424
25%      0.781250
50%      0.795321
75%      0.809524
max      0.868852
--- 480.9364721775055 seconds ---
'''

# In[]:

from sklearn import ensemble 
from sklearn.pipeline import Pipeline

start_time = time.time()
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=40, random_state=34)
adatree = ensemble.AdaBoostClassifier()
pca = PCA(n_components=10)
pipe = Pipeline(steps=[('pca', pca), ('tree', adatree)])
val_list = []
test_list=[]

for train_index, test_index in rskf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipe.fit(X_train, y_train)
    val_list.append(f1_score(y_test, pipe.predict(X_test)))
    test_list.append(f1_score(y_outside_test, pipe.predict(X_outside_test)))

plt.figure(figsize=(13, 5))
plt.plot(val_list, label='validate')
plt.plot(test_list, label='test')
plt.title('test and validate f1 score')
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.show()

val = pd.DataFrame(val_list).describe()
print('AdaBoost Classifier CV result 1', val)
tst = pd.DataFrame(test_list).describe()
print('AdaBoost Classifier test result 1', tst)
print("--- %s seconds ---" % (time.time() - start_time))

'''
AdaBoost Classifier CV result 1                 0
count  400.000000
mean     0.690079
std      0.096991
min      0.315789
25%      0.636364
50%      0.695652
75%      0.752155
max      0.916667
AdaBoost Classifier test result 1                 0
count  400.000000
mean     0.732709
std      0.031398
min      0.643478
25%      0.711584
50%      0.735043
75%      0.754098
max      0.819672
--- 46.24868297576904 seconds ---
'''

# In[]:



