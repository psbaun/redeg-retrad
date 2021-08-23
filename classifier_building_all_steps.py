#!/usr/bin/env python
# coding: utf-8

# ### Importing modules

# In[1]:


import pandas as pd
import numpy as np
import spacy
from textblob import TextBlob

from statistics import mean, stdev
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

from sklearn.preprocessing import binarize


# ### Loading Data

# In[2]:


data = pd.read_csv("project_18_dataset_combined.csv")
data = data[['label', 'text']]

#set display option
pd.set_option('display.max_colwidth', None)

#make target labels boolean
data['label']=data['label'].apply(lambda x: 1 if x == 14 else 0)


# In[3]:


print(data.shape)
print(data.label.value_counts())
data.head()


# ### Lemmatizing text

# In[4]:


nlp = spacy.load('en_core_web_md')
data['text_lemmatized'] = data['text'].apply(lambda x: " ".join([words.lemma_ for words in nlp(x)]))


# In[5]:


data.head()


# #### Q: WHAT SIZE EN_CORE_WEB TO USE???

# ### Evaluating effect of lemmatization (ceteris paribus)

# In[7]:


#train/test split on original and preprocessed data
X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(data.text, data.label, test_size=0.2, random_state=88, stratify=data.label)
X_train, X_test, y_train, y_test = train_test_split(data.text_lemmatized, data.label, test_size=0.2, random_state=88, stratify=data.label)


# In[8]:


print(y_train_old.value_counts())
print(y_train.value_counts())
print(y_test_old.value_counts())
print(y_test.value_counts())


# In[9]:


#vectorizing train data and transforming test data
vectorizer = CountVectorizer()
X_train_old_dtm = vectorizer.fit_transform(X_train_old)
X_test_old_dtm = vectorizer.transform(X_test_old)
X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)


# In[10]:


#train model on X_train_dtm
mnb_old = MultinomialNB()
mnb = MultinomialNB()
mnb_old.fit(X_train_old_dtm, y_train_old)
mnb.fit(X_train_dtm, y_train)


# In[11]:


#make class prediction for X_test_dtm
y_pred_class_old = mnb_old.predict(X_test_old_dtm)
y_pred_class = mnb.predict(X_test_dtm)


# In[12]:


from collections import Counter
print(Counter(y_pred_class_old))
print(Counter(y_pred_class))


# In[13]:


#comparing confusing matrices
print('old model cm:')
print(confusion_matrix(y_test_old, y_pred_class_old))
print('new model cm:')
print(confusion_matrix(y_test, y_pred_class))


# In[14]:


#calculating accuracy, precison
print('accuracy score old model:', accuracy_score(y_test_old, y_pred_class_old))
print('accuracy score new model:', accuracy_score(y_test, y_pred_class))
print('-----')
print('precision score old model:', precision_score(y_test_old, y_pred_class_old))
print('precision score new model:', precision_score(y_test, y_pred_class))
print('-----')
print('recall score old model:', recall_score(y_test_old, y_pred_class_old))
print('recall score new model:', recall_score(y_test, y_pred_class))


# ### Spelling correction

# In[15]:


#def spelling_corrector(txt):
#    blob = TextBlob(txt)
#    return str(blob.correct())


#data['text_spelling_corrected'] = data['text_lemmatized'].apply(lambda x : [spelling_corrector(x)])
#data.head()


# In[16]:


#data['text_spelling_corrected2'] = data['text_lemmatized'].apply(lambda x: ' '.join(TextBlob(x).correct()))
#data.head


# ### Evaluating effect of spell corrector

# In[ ]:





# ### Effect of random states (ceteris paribus)

# In[17]:


#make model with range of random states in train/test split
random_state_range = range(0, 1000)
rs_scores = []
for rs in random_state_range:
    vectorizer = CountVectorizer()
    mnb = MultinomialNB()
    X_train, X_test, y_train, y_test = train_test_split(data.text_lemmatized, data.label, random_state=rs, test_size=0.2, stratify=data.label)
    
    X_train_dtm = vectorizer.fit_transform(X_train)
    X_test_dtm = vectorizer.transform(X_test)
    
    mnb.fit(X_train_dtm, y_train)
    
    y_pred_class = mnb.predict(X_test_dtm)
    
    rs_scores.append(precision_score(y_test, y_pred_class))


# In[18]:


#calculating mean precision and standard deviation
print('mean precision:', mean(rs_scores))
print('st.dev. of mean prec:', stdev(rs_scores))

#make plot
plt.plot(random_state_range, rs_scores)
plt.xlabel('Random state value')
plt.ylabel('Testing precision')

plt.grid(True)


# ### Effect of test size (ceteris paribus)

# In[19]:


#make model with varying test sizes in train/test split
test_size_range = np.linspace(0.05,0.5,91)
test_size_scores = []

for ts in test_size_range:
    vectorizer = CountVectorizer()
    mnb = MultinomialNB()
    X_train, X_test, y_train, y_test = train_test_split(data.text_lemmatized, data.label, test_size=ts, random_state=88, stratify=data.label)
    
    X_train_dtm = vectorizer.fit_transform(X_train)
    X_test_dtm = vectorizer.transform(X_test)
    
    mnb.fit(X_train_dtm, y_train)
    
    y_pred_class = mnb.predict(X_test_dtm)
    
    test_size_scores.append(precision_score(y_test, y_pred_class))


# In[20]:


#calculating mean precision and standard deviation
print('mean precision:', mean(test_size_scores))
print('st.dev. of mean prec:', stdev(test_size_scores))

#make plot
plt.plot(test_size_range, test_size_scores)
plt.xlabel('Test size value')
plt.ylabel('Testing precision')

plt.grid(True)


# ### Hyperparameter tuning

# In[21]:


#train/test splitting
X_train, X_test, y_train, y_test = train_test_split(data.text_lemmatized, data.label, test_size=0.2, random_state=88, stratify=data.label)


# In[22]:


#making pipeline
pipeline = Pipeline([ ('vectorizer', CountVectorizer()), ('classifier', MultinomialNB()) ])


# ##### Q: include ('tfidf', TfidfTransformer()) in pipeline???

# In[23]:


#grid = {
    #'vectorizer__strip_accents': [None, 'ascii', 'unicode'], #works
    #'vectorizer__lowercase': [True, False], #works
    #'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)], #works
    #'vectorizer__stop_words': [None, 'english'], #works
    #'vectorizer__max_df': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], #works
    #'vectorizer__min_df': [1, 0.001, 0.002, 0.005, 0.01, 0.02], #works
    #'vectorizer__max_features': [None, 1, 10, 100, 1000, 10000], #works
    #'classifier__alpha': [0.1, 0.5, 1.0, 2.0], #works
    #'classifier__fit_prior': [True, False], #works
    #'classifier__class_prior': [[0.1, 0.9], [0.1, 0.8], [0.2, 0.9], [0.2, 0.8]], #works
    #'tfidf__norm': ['l1', 'l2'], #works
    #'tfidf__use_idf': [True, False], #works
    #'tfidf__smooth_idf': [True, False], #works
    #'tfidf__sublinear_tf': [True, False], #works
#}

#grid_search = GridSearchCV(pipeline, param_grid=grid, scoring='precision', cv=10)
#grid_search.fit(X, y)

#print("-----------")
#print(grid_search.best_score_)
#print(grid_search.best_params_)


# In[24]:


grid = {}
        

grid_search = GridSearchCV(pipeline, param_grid=grid, scoring='precision', cv=10)
grid_search.fit(data.text_lemmatized, data.label)

print(grid_search.best_score_)
print(grid_search.best_params_)

results = pd.DataFrame(grid_search.cv_results_)
results[['params', 'mean_test_score','std_test_score']]


# ### Evaluation between models with default and tuned parameters

# In[25]:


#calculating accuracy, precison and roc_auc between models with default and tuned parameters
#set best parameters in pipeline for comparison

pipeline_old = Pipeline([ ('vectorizer', CountVectorizer()), ('classifier', MultinomialNB()) ])
pipeline = Pipeline([ ('vectorizer', CountVectorizer()), ('classifier', MultinomialNB()) ])
     
model_old = pipeline_old.fit(X_train, y_train)
model = pipeline.fit(X_train, y_train)

y_pred_class_old = model_old.predict(X_test)
y_pred_class = model.predict(X_test)

print('accuracy score before tuning model:', accuracy_score(y_test, y_pred_class_old))
print('accuracy score after tuning:', accuracy_score(y_test, y_pred_class))
print('-----')
print('precision score before tuning:', precision_score(y_test, y_pred_class_old))
print('precision score after tuning:', precision_score(y_test, y_pred_class))
print('-----')
print('recall score before tuning:', recall_score(y_test, y_pred_class_old))
print('recall score after tuning:', recall_score(y_test, y_pred_class))


# In[26]:


print(Counter(y_test))
print(Counter(y_pred_class_old))
print(Counter(y_pred_class))


# In[27]:


#comparing confusing matrices
print('old model cm:')
print(confusion_matrix(y_test, y_pred_class_old))
print('new model cm:')
print(confusion_matrix(y_test, y_pred_class))


# #### Q: Crossvalidation on final model???

# ### Examining results

# In[28]:


#see false positive comments
false_positives = pd.DataFrame({'false_positives': X_test[(y_pred_class==1) & (y_test==0)]})
print(false_positives.shape)
false_positives


# In[29]:


#see false negative comments
false_negatives = pd.DataFrame({'false_negatives': X_test[(y_pred_class==0) & (y_test==1)]})
print(false_negatives.shape)
false_negatives


# In[30]:


# store the vocabulary of X_train
X_train_tokens = vectorizer.get_feature_names()

# number of times each token appears across all history comments
hisco_token_count = mnb.feature_count_[1, :]

# number of times each token appears across all non-history comments
nhisco_token_count = mnb.feature_count_[0, :]


# In[31]:


# create a DataFrame of tokens with their separate his and non-his counts
tokens = pd.DataFrame({'token':X_train_tokens, 'non_history':nhisco_token_count, 'history':hisco_token_count}).set_index('token')

# add 1 to each label counts to avoid dividing by 0
tokens['non_history'] = tokens.non_history + 1
tokens['history'] = tokens.history + 1

# convert his and non-his counts into frequencies
tokens['non_history'] = tokens.non_history / mnb.class_count_[0]
tokens['history'] = tokens.history / mnb.class_count_[1]

# calculate the ratio of his-to-non-his for each token
tokens['history_ratio'] = tokens.history / tokens.non_history

# calculate the ratio of non-his-to-his for each token
tokens['non_history_ratio'] = tokens.non_history / tokens.history


# In[32]:


# examine the DataFrame sorted by history_comments_ratio
tokens_his = tokens.sort_values('history_ratio', ascending=False)
tokens_his[0:10]


# In[33]:


# examine the DataFrame sorted by non_history_comments_ratio
tokens_non_his = tokens.sort_values('non_history_ratio', ascending=False)
tokens_non_his[0:10]


# In[34]:


# look up the history_ratio for a given token
# tokens.loc['', 'history_ratio']


# ### Threshold and further evaluation

# In[35]:


y_pred_prob = model.predict_proba(X_test)[:, 1]


# In[36]:


plt.hist(y_pred_prob, bins=50)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of history-comment')
plt.ylabel('Frequency')

plt.grid(True)


# In[37]:


y_pred_class_new = binarize([y_pred_prob], 0.9)[0]


# In[38]:


print(confusion_matrix(y_test, y_pred_class))
print(confusion_matrix(y_test, y_pred_class_new))


# In[39]:


print(accuracy_score(y_test, y_pred_class))
print(accuracy_score(y_test, y_pred_class_new))
print('-----')
print(precision_score(y_test, y_pred_class))
print(precision_score(y_test, y_pred_class_new))
print('-----')
print(recall_score(y_test, y_pred_class))
print(recall_score(y_test, y_pred_class_new))


# In[40]:


threshold_range = (np.linspace(0.01,0.99,99))
y_pred_class_precision_scores = []
y_pred_class_recall_scores = []

for th in threshold_range:
    #insert model also?
    
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred_class = binarize([y_pred_prob], th)[0]
    
    y_pred_class_precision_scores.append(precision_score(y_test, y_pred_class))
    y_pred_class_recall_scores.append(recall_score(y_test, y_pred_class))


# In[41]:


fig, ax1 = plt.subplots()

color = 'tab:orange'
ax1.set_xlabel('threshold value')
ax1.set_ylabel('recall score', color=color)
ax1.plot(threshold_range, y_pred_class_recall_scores, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('precision score', color=color)
ax2.plot(threshold_range, y_pred_class_precision_scores, color=color)
ax2.tick_params(axis='y', labelcolor=color)


# In[42]:


#ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, color='orange')
plt.plot([0,1],[0,1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for history-comment classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[43]:


#precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
plt.plot(precision, recall, color='orange')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('precision-recall curve for history-comment classifier')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid(True)


# In[44]:


#ROC-AUC score
print(roc_auc_score(y_test, y_pred_prob))


# # Repeating steps with balanced data

# ### Loading original data and balancing

# In[45]:


data = pd.read_csv("project_18_dataset_combined.csv")
data = data[['label', 'text']]

#set display option
pd.set_option('display.max_colwidth', None)

#make target labels boolean
data['label']=data['label'].apply(lambda x: 1 if x == 14 else 0)


# In[46]:


print(data.shape)
print(data.label.value_counts())
data.head()


# In[47]:


#Balancing data
count_label_0, count_label_1 = data.label.value_counts()

data_label_1 = data[data['label'] == 1]
data_label_0 = data[data['label'] == 0]

data_label_0_b = data_label_0.sample(count_label_1, random_state=88)
data_b = pd.concat([data_label_0_b, data_label_1])
print(data_b.shape)
print(data_b.label.value_counts())
data_b.head()


# ### Lemmatizing and evaluating effect of lemmatization on balanced data

# In[48]:


#lemmatizing
nlp = spacy.load('en_core_web_md')
data_b['text_lemmatized'] = data_b['text'].apply(lambda x: " ".join([words.lemma_ for words in nlp(x)]))


# In[49]:


#train/test split on original and preprocessed data
X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(data_b.text, data_b.label, test_size=0.2, random_state=88, stratify=data_b.label)
X_train, X_test, y_train, y_test = train_test_split(data_b.text_lemmatized, data_b.label, test_size=0.2, random_state=88, stratify=data_b.label)


# In[50]:


#vectorizing train data and transforming test data
vectorizer = CountVectorizer()
X_train_old_dtm = vectorizer.fit_transform(X_train_old)
X_test_old_dtm = vectorizer.transform(X_test_old)
X_train_dtm = vectorizer.fit_transform(X_train)
X_test_dtm = vectorizer.transform(X_test)


# In[51]:


#train model on X_train_dtm
mnb_old = MultinomialNB()
mnb = MultinomialNB()
mnb_old.fit(X_train_old_dtm, y_train_old)
mnb.fit(X_train_dtm, y_train)


# In[52]:


#make class prediction for X_test_dtm
y_pred_class_old = mnb_old.predict(X_test_old_dtm)
y_pred_class = mnb.predict(X_test_dtm)


# In[53]:


#comparing confusing matrices
print('old model cm:')
print(confusion_matrix(y_test_old, y_pred_class_old))
print('new model cm:')
print(confusion_matrix(y_test, y_pred_class))


# In[54]:


#calculating accuracy, precison
print('accuracy score old model:', accuracy_score(y_test_old, y_pred_class_old))
print('accuracy score new model:', accuracy_score(y_test, y_pred_class))
print('-----')
print('precision score old model:', precision_score(y_test_old, y_pred_class_old))
print('precision score new model:', precision_score(y_test, y_pred_class))
print('-----')
print('recall score old model:', recall_score(y_test_old, y_pred_class_old))
print('recall score new model:', recall_score(y_test, y_pred_class))


# ### Spelling correction and evaluating effect of spell corrector on balanced data

# In[55]:


#def spelling_corrector(txt):
#    blob = TextBlob(txt)
#    return str(blob.correct())


#data_b['text_spelling_corrected'] = data_b['text_lemmatized'].apply(lambda x : [spelling_corrector(x)])
#data_b.head()


# In[56]:


#data_b['text_spelling_corrected2'] = data_b['text_lemmatized'].apply(lambda x: ' '.join(TextBlob(x).correct()))
#data_b.head


# ### Effect of random states on balanced data (ceteris paribus)

# In[57]:


#make model with range of random states in train/test split

random_state_range = range(0, 1000)
rs_scores = []
for rs in random_state_range:
    vectorizer = CountVectorizer()
    mnb = MultinomialNB()
    X_train, X_test, y_train, y_test = train_test_split(data_b.text_lemmatized, data_b.label, random_state=rs, test_size=0.2, stratify=data_b.label)
    
    X_train_dtm = vectorizer.fit_transform(X_train)
    X_test_dtm = vectorizer.transform(X_test)
    
    mnb.fit(X_train_dtm, y_train)
    
    y_pred_class = mnb.predict(X_test_dtm)
    
    rs_scores.append(precision_score(y_test, y_pred_class))


# In[58]:


#calculating mean precision and standard deviation
print('mean precision:', mean(rs_scores))
print('st.dev. of mean prec:', stdev(rs_scores))

#make plot
plt.plot(random_state_range, rs_scores)
plt.xlabel('Random state value')
plt.ylabel('Testing precision')

plt.grid(True)


# ### Effect of test sizes on balanced data (ceteris paribus)

# In[59]:


#make model with varying test sizes in train/test split
test_size_range = np.linspace(0.05,0.5,91)
test_size_scores = []

for ts in test_size_range:
    vectorizer = CountVectorizer()
    mnb = MultinomialNB()
    X_train, X_test, y_train, y_test = train_test_split(data_b.text_lemmatized, data_b.label, test_size=ts, random_state=88, stratify=data_b.label)
    
    X_train_dtm = vectorizer.fit_transform(X_train)
    X_test_dtm = vectorizer.transform(X_test)
    
    mnb.fit(X_train_dtm, y_train)
    
    y_pred_class = mnb.predict(X_test_dtm)
    
    test_size_scores.append(precision_score(y_test, y_pred_class))


# In[60]:


#calculating mean precision and standard deviation
print('mean precision:', mean(test_size_scores))
print('st.dev. of mean prec:', stdev(test_size_scores))

#make plot
plt.plot(test_size_range, test_size_scores)
plt.xlabel('Test size value')
plt.ylabel('Testing precision')

plt.grid(True)


# ### Hyperparameter tuning on balanced data

# In[61]:


#train/test splitting
X_train, X_test, y_train, y_test = train_test_split(data_b.text_lemmatized, data_b.label, test_size=0.2, random_state=88, stratify=data_b.label)


# In[62]:


#making pipeline
pipeline = Pipeline([ ('vectorizer', CountVectorizer()), ('classifier', MultinomialNB()) ])


# #### Q: include ('tfidf', TfidfTransformer()) in pipeline???

# In[63]:


#grid = {
    #'vectorizer__strip_accents': [None, 'ascii', 'unicode'], #works
    #'vectorizer__lowercase': [True, False], #works
    #'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)], #works
    #'vectorizer__stop_words': [None, 'english'], #works
    #'vectorizer__max_df': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], #works
    #'vectorizer__min_df': [1, 0.001, 0.002, 0.005, 0.01, 0.02], #works
    #'vectorizer__max_features': [None, 1, 10, 100, 1000, 10000], #works
    #'classifier__alpha': [0.1, 0.5, 1.0, 2.0], #works
    #'classifier__fit_prior': [True, False], #works
    #'classifier__class_prior': [[0.1, 0.9], [0.1, 0.8], [0.2, 0.9], [0.2, 0.8]], #works
    #'tfidf__norm': ['l1', 'l2'], #works
    #'tfidf__use_idf': [True, False], #works
    #'tfidf__smooth_idf': [True, False], #works
    #'tfidf__sublinear_tf': [True, False], #works
#}

#grid_search = GridSearchCV(pipeline, param_grid=grid, scoring='precision', cv=10)
#grid_search.fit(X, y)

#print("-----------")
#print(grid_search.best_score_)
#print(grid_search.best_params_)


# In[64]:


grid = {}
        

grid_search = GridSearchCV(pipeline, param_grid=grid, scoring='precision', cv=10)
grid_search.fit(data_b.text_lemmatized, data_b.label)

print(grid_search.best_score_)
print(grid_search.best_params_)

results = pd.DataFrame(grid_search.cv_results_)
results[['params', 'mean_test_score','std_test_score']]


# ### Evaluation between models with default and tuned parameters (balanced data)

# In[65]:


#calculating accuracy, precison and roc_auc between models with default and tuned parameters
#set best parameters in pipeline for comparison

pipeline_old = Pipeline([ ('vectorizer', CountVectorizer()), ('classifier', MultinomialNB()) ])
pipeline = Pipeline([ ('vectorizer', CountVectorizer()), ('classifier', MultinomialNB()) ])
     
model_old = pipeline_old.fit(X_train, y_train)
model = pipeline.fit(X_train, y_train)

y_pred_class_old = model_old.predict(X_test)
y_pred_class = model.predict(X_test)

print('accuracy score before tuning model:', accuracy_score(y_test, y_pred_class_old))
print('accuracy score after tuning:', accuracy_score(y_test, y_pred_class))
print('-----')
print('precision score before tuning:', precision_score(y_test, y_pred_class_old))
print('precision score after tuning:', precision_score(y_test, y_pred_class))
print('-----')
print('recall score before tuning:', recall_score(y_test, y_pred_class_old))
print('recall score after tuning:', recall_score(y_test, y_pred_class))


# In[66]:


print(Counter(y_test))
print(Counter(y_pred_class_old))
print(Counter(y_pred_class))


# In[67]:


#comparing confusing matrices
print('old model cm:')
print(confusion_matrix(y_test, y_pred_class_old))
print('new model cm:')
print(confusion_matrix(y_test, y_pred_class))


# #### Q: Crossvalidation on final model???

# ### Examining results (balanced data)

# In[68]:


#see false positive comments
false_positives = pd.DataFrame({'false_positives': X_test[(y_pred_class==1) & (y_test==0)]})
print(false_positives.shape)
false_positives


# In[69]:


#see false negative comments
false_negatives = pd.DataFrame({'false_negatives': X_test[(y_pred_class==0) & (y_test==1)]})
print(false_negatives.shape)
false_negatives


# In[70]:


# store the vocabulary of X_train
X_train_tokens = vectorizer.get_feature_names()

# number of times each token appears across all history comments
hisco_token_count = mnb.feature_count_[1, :]

# number of times each token appears across all non-history comments
nhisco_token_count = mnb.feature_count_[0, :]


# In[71]:


# create a DataFrame of tokens with their separate his and non-his counts
tokens = pd.DataFrame({'token':X_train_tokens, 'non_history':nhisco_token_count, 'history':hisco_token_count}).set_index('token')

# add 1 to each label counts to avoid dividing by 0
tokens['non_history'] = tokens.non_history + 1
tokens['history'] = tokens.history + 1

# convert his and non-his counts into frequencies
tokens['non_history'] = tokens.non_history / mnb.class_count_[0]
tokens['history'] = tokens.history / mnb.class_count_[1]

# calculate the ratio of his-to-non-his for each token
tokens['history_ratio'] = tokens.history / tokens.non_history

# calculate the ratio of non-his-to-his for each token
tokens['non_history_ratio'] = tokens.non_history / tokens.history


# In[72]:


# examine the DataFrame sorted by history_comments_ratio
tokens_his = tokens.sort_values('history_ratio', ascending=False)
tokens_his[0:10]


# In[73]:


# examine the DataFrame sorted by non_history_comments_ratio
tokens_non_his = tokens.sort_values('non_history_ratio', ascending=False)
tokens_non_his[0:10]


# In[74]:


# look up the history_ratio for a given token
#tokens.loc['', 'history_ratio']


# ### Threshold and further evaluation (balanced data)

# In[75]:


y_pred_prob = model.predict_proba(X_test)[:, 1]


# In[76]:


plt.hist(y_pred_prob, bins=50)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of history-comment')
plt.ylabel('Frequency')

plt.grid(True)


# In[77]:


y_pred_class_new = binarize([y_pred_prob], 0.9)[0]


# In[78]:


print(confusion_matrix(y_test, y_pred_class))
print(confusion_matrix(y_test, y_pred_class_new))


# In[79]:


print('default threshold accuracy:', accuracy_score(y_test, y_pred_class))
print('new threshold accuracy:', accuracy_score(y_test, y_pred_class_new))
print('-----')
print('default threshold precision:', precision_score(y_test, y_pred_class))
print('new threshold precision:', precision_score(y_test, y_pred_class_new))
print('-----')
print('default threshold recall:', recall_score(y_test, y_pred_class))
print('new threshold recall:', recall_score(y_test, y_pred_class_new))


# In[80]:


threshold_range = (np.linspace(0.01,0.99,99))
y_pred_class_precision_scores = []
y_pred_class_recall_scores = []

for th in threshold_range:
    #insert model also?
    
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred_class_new = binarize([y_pred_prob], th)[0]
    
    y_pred_class_precision_scores.append(precision_score(y_test, y_pred_class_new))
    y_pred_class_recall_scores.append(recall_score(y_test, y_pred_class_new))


# In[82]:


fig, ax1 = plt.subplots()

color = 'tab:orange'
ax1.set_xlabel('threshold value')
ax1.set_ylabel('recall score', color=color)
ax1.plot(threshold_range, y_pred_class_recall_scores, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('precision score', color=color)
ax2.plot(threshold_range, y_pred_class_precision_scores, color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.grid(True)


# In[83]:


#ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, color='orange')
plt.plot([0,1],[0,1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for history-comment classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[84]:


#precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
plt.plot(precision, recall, color='orange')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('precision-recall curve for history-comment classifier')
plt.xlabel('Recall') #or is it reverse????
plt.ylabel('Precision')
plt.grid(True)


# In[85]:


#ROC-AUC score
print(roc_auc_score(y_test, y_pred_prob))


# #### Q: possible to import batch of his-comments?

# #### Q: Pipeline for content analysis?
