import pandas as pd
path = 'Rand Sentiment Analysis Dataset.csv'
sms = pd.read_csv(path, error_bad_lines=False, encoding = "ISO-8859-1")
print(sms.shape)
print(sms.head(10))
print(sms.Sentiment.value_counts())

X = sms.SentimentText
y = sms.Sentiment
print(X.shape)
print(y.shape)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
Xtfid = vectorizer.fit_transform(X)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xtfid, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn import preprocessing
X_train_norm = preprocessing.normalize(X_train, norm='l2')
X_test_norm = preprocessing.normalize(X_test, norm='l2')

from sklearn import svm
clf = svm.SVC(gamma='auto', kernel='linear')
clf.fit(X_train_norm, y_train)
y_pred_class = clf.predict(X_test_norm)


import numpy
numpy.set_printoptions(threshold=numpy.nan)

from sklearn import metrics
print(clf.score(X_test, y_test))
y_score = clf.decision_function(X_test)
print(metrics.accuracy_score(y_test, y_pred_class))
print(metrics.roc_auc_score(y_test, y_score))
fpr, tpr, thresh = (metrics.roc_curve(y_test, y_score))
roc_auc = metrics.auc(fpr, tpr)
#print(metrics.confusion_matrix(y_test,y_pred_class))

import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
