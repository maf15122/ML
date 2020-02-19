#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt


# In[5]:


pwd


# In[6]:


import os


# In[7]:


os.chdir('C:\\Users\\ancha\\OneDrive\\Desktop')


# In[8]:


pwd


# In[9]:


userdemo = pd.read_excel('C:\\Users\\ancha\\OneDrive\\Desktop\\u.info.xlsx')


# In[10]:


userdemo.head()


# In[11]:


columns_rating = ['UserID', 'MovieID', 'Rating']
userinfo = pd.read_table('C:\\Users\\ancha\\OneDrive\\Desktop\\userinfo.txt',names=columns_rating, usecols=range(3), encoding="ISO-8859-1")


# In[12]:


userinfo.head()


# In[13]:


merged=pd.merge(userdemo, userinfo)


# In[14]:


merged.head()


# In[15]:


genre = pd.read_excel('C:\\Users\\ancha\\OneDrive\\Desktop\\u.genre.xlsx')


# In[16]:


genre.head()


# In[17]:


mergedgenre=pd.merge(merged,genre)


# In[18]:


mergedgenre.head()


# In[19]:


pip install pymf3 


# In[20]:


from sklearn.preprocessing import StandardScaler


# In[47]:


features = ['UserID', 'Age', 'MovieID', 'Rating']
x = mergedgenre.loc[:, features].values
y = mergedgenre.loc[:,['Gender','Occupation']].values
x = StandardScaler().fit_transform(x)


# In[53]:


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, mergedgenre[['Gender','Rating']]], axis = 1)
finalDf.head()


# In[54]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
Gender = ['M', 'F']
colors = ['r', 'g', 'b']
for Gender, color in zip(Gender,colors):
    indicesToKeep = finalDf['Gender'] == Gender
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(Gender)
ax.grid()


# In[63]:


newdata=merged.drop(['ZipCode','Occupation'],axis=1)
newdata.head()


# In[95]:


features = ['UserID', 'Age', 'MovieID', 'Rating']
x = newdata.loc[:, features].values
y = newdata.loc[:,['Gender']].values
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, mergedgenre[['Gender','Rating']]], axis = 1)
finalDf.head()


# In[96]:


from sklearn.preprocessing import StandardScaler


# In[106]:


newdata1=newdata.drop(['Gender'],axis=1)
newdata2=merged.drop(['UserID','Age','MovieID','Rating','Occupation','ZipCode'],axis=1)
scaler = StandardScaler()
scaler.fit(newdata1)


# In[98]:


scaled_data = scaler.transform(newdata1)


# In[99]:


from sklearn.decomposition import PCA


# In[100]:


pca = PCA(n_components=2)
pca.fit(scaled_data)


# In[101]:


x_pca = pca.transform(scaled_data)


# In[102]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=newdata1['MovieID'],cmap='rainbow')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[108]:


newdata1.head()


# In[111]:


from sklearn.preprocessing import StandardScaler


# In[114]:


#Resacle the feature vector
X=newdata1[['UserID','Age','MovieID','Rating']]
newdata1=StandardScaler().fit_transform(X) 


# In[115]:


newdata1


# In[119]:


#Now we wanna know the Covariance matrix using the feature vector
features= newdata1.T
covariance_matrix=np.cov(features)
print(covariance_matrix)


# In[122]:


#now we want the eigenvalues and eigenvectors of this covariance matrix
eig_vals, eig_vecs=np.linalg.eig(covariance_matrix)
print('Eigenvalues \n%s' %eig_vecs)


# In[123]:


print('Eigenvalues \n%s' %eig_vals)


# In[132]:


eig_vals[0]/sum(eig_vals)


# In[133]:


projected_x=newdata1.dot(eig_vecs.T[0])


# In[134]:


projected_x


# In[138]:





# In[139]:


newdata2=merged.drop(['ZipCode','Occupation','UserID','Age','MovieID','Rating'],axis=1)


# In[140]:


result=pd.DataFrame(projected_x, columns=['PC1'])
result['y-axis']=0.0
result['label']=newdata2


# In[141]:


result.head(10)


# In[144]:


import seaborn as sns
sns.lmplot('PC1','y-axis',data=result,fit_reg=False, scatter_kws={"s":50},hue='label')
plt.title('PCA Result')


# In[145]:


from sklearn import decomposition


# In[148]:


pca=decomposition.PCA(n_components=2)
sklearn_pca_x=pca.fit_transform(newdata1)


# In[155]:


sklearn_result=pd.DataFrame(sklearn_pca_x,columns=['PC1','PC2'])
sklearn_result['y-axis']=0.0
sklearn_result['label']=newdata2
sns.lmplot('PC1','PC2',data=sklearn_result,fit_reg=False,scatter_kws={"s":50},hue="label")


# In[ ]:




