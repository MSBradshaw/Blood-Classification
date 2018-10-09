import Classification_Utils as cu
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('blood-samples.tsv', sep='\t', index_col='id')

#imputed missing values with 0
df = df.fillna(0)
print(df.shape)

#transform data and then remove extra information not needed for training
clean_df = df.T.iloc[2:-8,:]

#create labels
extract_label = lambda x: re.sub('_.*', '', x)
labels = list(map(extract_label,clean_df.index.tolist()))


train_df, test_df, train_labels, test_labels = train_test_split(
    clean_df, labels, 
    test_size=0.30,    # 30% of the data held out in test set
    random_state=0,    # Setting random_state ensures the same train/test split occurs each time this is run
    stratify=labels)   # Maintain ratio of tissues represented in each set

train_features = train_df.columns.values.tolist()


NUM_SPLITS = 100 # number of train/test splits in cross validation

start = time.time()
knn = cu.knn_model_crossval(train_df, train_labels, NUM_SPLITS)
end = time.time()
print("KNN Runtime:", (end - start)/60, "minutes")

start = time.time()
mlp = cu.mlp_crossval(train_df, train_labels, NUM_SPLITS)
end = time.time()
print("MLP Runtime:", (end - start)/60, "minutes")

start = time.time()
lr = cu.logistic_regression_model_crossval(train_df, train_labels, NUM_SPLITS)
end = time.time()
print("LR Runtime:", (end - start)/60, "minutes")

start = time.time()
gnb = cu.bayes_gaussian_model_crossval(train_df, train_labels, NUM_SPLITS)
end = time.time()
print("Gaussian NB Runtime:", (end - start)/60, "minutes")


start = time.time()
svc = cu.SVC_model_crossval(train_df, train_labels, NUM_SPLITS)
end = time.time()
print("SVC Runtime:", (end - start)/60, "minutes")

start = time.time()
rf = cu.randomforest_model_crossval(train_df, train_labels, NUM_SPLITS)
end = time.time()
print("RF Runtime:", (end - start)/60, "minutes")

start = time.time()
gbc = cu.gradient_boosting_crossval(train_df, train_labels, NUM_SPLITS)
end = time.time()
print("Gradient Boosting Runtime:", (end - start)/60, "minutes")

lr_pred = lr.predict(test_df)
lr_result = lr.score(test_df, test_labels)


rf_pred = rf.predict(test_df)
rf_result = rf.score(test_df, test_labels)

svc_pred = svc.predict(test_df)
svc_result = svc.score(test_df, test_labels)

gbc_pred = gbc.predict(test_df)
gbc_result = gbc.score(test_df, test_labels)

gnb_pred = gnb.predict(test_df)
gnb_result = gnb.score(test_df, test_labels)

knn_pred = knn.predict(test_df)
knn_result = knn.score(test_df, test_labels)

mlp_pred = mlp.predict(test_df)
mlp_result = mlp.score(test_df, test_labels)

print(lr_result)
print(rf_result)
print(svc_result)
print(gbc_result)
print(gnb_result)
print(knn_result)
print(mlp_result)

#PCA graph generation

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(clean_df)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

principalDf['label'] = labels


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1 ' + str(pca.explained_variance_ratio_[0]), fontsize = 15)
ax.set_ylabel('Principal Component 2 ' + str(pca.explained_variance_ratio_[1]), fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['M', 'B']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = principalDf['label'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
               , principalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
fig.savefig('blood-PCA-plot.png')

pca.explained_variance_ratio_[]

