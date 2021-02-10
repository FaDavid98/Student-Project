import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import set_printoptions
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler  
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel

np.random.seed(8)
tf.random.set_seed(10)
#loading file
df = pd.read_csv ('student-mat.csv',delimiter=';')

#converting binaric features
dummy=pd.get_dummies(df[['sex','school','address','famsize','Pstatus','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']])

df2=pd.concat((df,dummy),axis=1)
df2=df2.drop(['sex','school','address','famsize','Pstatus','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic',],axis=1)
df2=df2.drop(['sex_M','school_MS','internet_no','romantic_no','paid_no','activities_no','nursery_no','higher_no','schoolsup_no','famsup_no','Pstatus_A','famsize_LE3','address_U'],axis=1)
df2=df2.rename(columns={'sex_F':'sex','school_GP':'school','internet_yes':'internet','romantic_yes':'romantic','paid_yes':'paid','activities_yes':'activities',
'nursery_yes':'nursery','higher_yes':'higher','schoolsup_yes':'schoolsup','famsup_yes':'famsup','Pstatus_T':'Pstatus','famsize_GT3':'famsize','address_R':'address'})

#moving G-s after features
cols_to_move = ['G1', 'G2' ,'G3']
df2 = df2[[ col for col in df.columns if col not in cols_to_move ] +cols_to_move] 

#converting nominal features:
l1=LabelEncoder()

# converting Mjob:
l1.fit(df2['Mjob'])
df2.Mjob=l1.transform(df2.Mjob)

#converting Fjob:
l1.fit(df2['Fjob'])
df2.Fjob=l1.transform(df2.Fjob)

#converting reason 
l1.fit(df2['reason'])
df2.reason=l1.transform(df2.reason)

#gconverting uardian 
l1.fit(df2['guardian'])
df2.guardian=l1.transform(df2.guardian)

#descending classes
'''
df2['G3'] = df2['G3'].replace([1,2,3,4,5],1)
df2['G3'] = df2['G3'].replace([6,7,8,9,10],2)
df2['G3'] = df2['G3'].replace([11,12,13,14,15],3)
df2['G3'] = df2['G3'].replace([16,17,18,19,20],4)
'''

df2['G3'] = np.where(df2['G3'].between(0,5), 0, df2['G3'])
df2['G3'] = np.where(df2['G3'].between(6,10), 1, df2['G3'])
df2['G3'] = np.where(df2['G3'].between(11,15), 2, df2['G3'])
df2['G3'] = np.where(df2['G3'].between(16,20), 3, df2['G3'])

df2['G1'] = np.where(df2['G1'].between(0,5), 0, df2['G1'])
df2['G1'] = np.where(df2['G1'].between(6,10), 1, df2['G1'])
df2['G1'] = np.where(df2['G1'].between(11,15), 2, df2['G1'])
df2['G1'] = np.where(df2['G1'].between(16,20), 3, df2['G1'])

df2['G2'] = np.where(df2['G2'].between(0,5), 0, df2['G2'])
df2['G2'] = np.where(df2['G2'].between(6,10), 1, df2['G2'])
df2['G2'] = np.where(df2['G2'].between(11,15), 2, df2['G2'])
df2['G2'] = np.where(df2['G2'].between(16,20), 3, df2['G2'])

#x y selection
x=df2.iloc[:,0:32]
y=df2.iloc[:,-1]

#x normalization
names=x.columns
min_max_scaler = preprocessing.MinMaxScaler()
x = pd.DataFrame(min_max_scaler.fit_transform(x), columns=names)

#feature selection
'''
#chi2 selection
chi2_features = SelectKBest(chi2, k = 5) 
X_kbest_features = chi2_features.fit_transform(x, y) 
'''
'''
#RFE selection
rfe_selector=RFE(LogisticRegression(max_iter=10000), n_features_to_select=5)
X_kbest_features=rfe_selector.fit_transform(x,y)
'''
'''
#Select from model (LASSO)
embeded_lr_selector = SelectFromModel(LogisticRegression(max_iter= 1000),max_features=6)
X_kbest_features = embeded_lr_selector.fit_transform(x,y)
'''

#Select from model (RandomForest)
embeded_lr_selector = SelectFromModel(RandomForestClassifier(n_estimators=100),max_features=5)
X_kbest_features = embeded_lr_selector.fit_transform(x,y)


#number of features
m = X_kbest_features.shape[1]

#y categorizing
y=to_categorical(y)


#train test split
XTraining, XTest, YTraining, YTest = train_test_split(X_kbest_features, y, test_size=0.3)

#model 
model = tf.keras.models.Sequential()
opt = tf.keras.optimizers.Adam(0.006)
model.add(tf.keras.layers.Dense(16, input_shape=(m,), activation='relu'))
#model.add(tf.keras.layers.Dense(8, activation='relu'))
#model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(4 ,activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
history = model.fit(XTraining, YTraining, epochs=100, validation_split=0.2)

loss, acc = model.evaluate(XTest, YTest, verbose=2)
print('%.2f' % (acc*100))

#plotting loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

'''
#DecisionTree
clf=DecisionTreeClassifier()

clf=clf.fit(XTraining,YTraining)
y_pred=clf.predict(XTest)

print('Accuracy:', metrics.accuracy_score(YTest,y_pred)*100)
'''
'''
#RandomForest
clf=RandomForestClassifier(n_estimators=20)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(XTraining,YTraining)

y_pred=clf.predict(XTest)
print('Accuracy:',metrics.accuracy_score(YTest, y_pred)*100)
'''
'''
#LogistricRegression (first drop y_to categorical )

classifier = LogisticRegression(max_iter=800)
classifier.fit(XTraining, YTraining)
y_pred = classifier.predict(XTest)

print('Accuracy:',metrics.accuracy_score(YTest, y_pred)*100)
'''
