
# coding: utf-8

# In[40]:

import sqlalchemy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV

get_ipython().magic(u'matplotlib inline')


# In[29]:

engine = sqlalchemy.create_engine('postgresql://alexandertam@localhost/postgres')
data = pd.read_sql('SELECT * FROM data',con = engine)
data = data.dropna()


# In[34]:

try:
    del data["index"]
except:
    pass
y = data.iloc[:,1]
X = data.iloc[:,7:]


# ## Logistic Regression

# In[35]:

logiReg = LogisticRegression()
model = logiReg.fit(X,y)
predictions = model.predict(X)
print "Model score: ", model.score(X,y)


# In[36]:

print "Accuracy score: ", accuracy_score(y,predictions)
print "Precision score: ", precision_score(y,predictions)
print "Recall score: ", recall_score(y,predictions)


# In[37]:

pd.crosstab(y,predictions)


# In[51]:

def PredictProbaLogisticRegression(X_test,y_test,model):
    #print "Model score: ", model.score(X_test,y_test)
    return model.predict_proba(X_test)

proba_predictions = PredictProbaLogisticRegression(X,y,model)
#print proba_predictions


# In[87]:

def Classifier(probability,threshold):
    if(probability > threshold):
        return 1
    else:
        return 0
    
def ClassifyProbabilities(probabilities,threshold):
    classifieds = []
    for probability in probabilities:
        classifieds.append(Classifier(probability, threshold))
    return classifieds

def CriticalSuccessIndex(hits, misses, falseAlarms):
    return hits/float(hits + misses + falseAlarms)

def ActualToPredictedConfusionMatrix(y, predictions):
    actuals = pd.Series(y,name="Actual")
    predicted = pd.Series(predictions,name = "Predictions")
    return pd.crosstab(actuals,predicted)
    
newClassifieds = ClassifyProbabilities(pd.DataFrame(proba_predictions)[1],.1)
#print newClassifieds

ActualToPredictedConfusionMatrix(y,newClassifieds)
print "Model score: ", model.score(X,newClassifieds)
print "Accuracy score: ", accuracy_score(y,newClassifieds)
print "Precision score: ", precision_score(y,newClassifieds)
print "Recall score: ", recall_score(y,newClassifieds)

cm = ActualToPredictedConfusionMatrix(y,newClassifieds)
hits = cm[1][1]
falseAlarms = cm[1][0]
misses = cm[0][1]

print "Critical Success Index: ", CriticalSuccessIndex(hits,misses,falseAlarms)
cm


# ## Decision Tree

# In[42]:

dt = DecisionTreeClassifier(random_state=7)
model.fit(X,y)
predictions = model.predict(X)
print "Model score: ", model.score(X,y)


# In[44]:

print "Accuracy score: ", accuracy_score(y,predictions)
print "Precision score: ", precision_score(y,predictions)
print "Recall score: ", recall_score(y,predictions)


# In[ ]:



