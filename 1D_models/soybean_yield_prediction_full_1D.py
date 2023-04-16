#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.getcwd())


# In[3]:


## Import necessary packages
import numpy as np
from numpy import mean, std 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches 
import matplotlib.font_manager
import seaborn as sns
from osgeo import gdal
import os 
import glob
from PIL import Image
from pathlib import Path
import imagesize
import scipy.stats as ss
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score
import skimage
from skimage.transform import resize
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten,BatchNormalization,Input,ZeroPadding2D, Add
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint,LearningRateScheduler
from keras_preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam


# In[197]:


## Import the soybean yield csv
import pandas as pd
yield_df = pd.read_csv(r'D:\soybean_yield\csv\yield_result_stacks_gold.csv')


# In[198]:


## Display dataframe to get a sense of the whole dataset
yield_df


# In[199]:


## Round dataframe to two decimals places
yield_df=yield_df.round(2)
yield_df


# In[200]:


# plotting histogram and density
## Display yield data via histogram
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(16, 8))
sns.distplot(a=yield_df['yield_kgha'], color='red',hist_kws={"color":"green","edgecolor": 'black'})
plt.title('Soybean Yield', fontsize=20);
plt.ylabel('Density', fontsize=16)
plt.xlabel('Soybean Yield (kg/ha)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)	
plt.grid(True)
plt.savefig(r'D:\soybean_yield\figures\yield_distribution_1.png', dpi=300,bbox_inches='tight')
plt.show()


# In[201]:


## Shuffle the data to get a random sample
data_shuffled=yield_df.sample(frac=1).reset_index(drop=True)
data_shuffled


# In[202]:


## Check if NANs exist
data_shuffled.isnull().values.any()


# In[203]:


## Create a list of .tif files
chips = []
os.chdir('D:/soybean_yield/macro_corpus/training_corpus_stacked/')
for file in glob.glob('*.tif'):
    chips.append(file)

# Identify Image Resolutions
root='D:/soybean_yield/test_folder_aug/'
# Get the Image Resolutions
imgs = [img.name for img in Path(root).iterdir() if img.suffix == ".tif"]
img_meta = {}
for f in imgs: img_meta[str(f)] = imagesize.get(root+f)

# Convert it to Dataframe and compute aspect ratio
img_meta_df = pd.DataFrame.from_dict([img_meta]).T.reset_index().set_axis(['FileName', 'Size'], axis='columns', inplace=False)
img_meta_df[["Height", "Width"]] = pd.DataFrame(img_meta_df["Size"].tolist(), index=img_meta_df.index)
img_meta_df["Aspect Ratio"] = round(img_meta_df["Height"] / img_meta_df["Width"], 2)

print(f'Total Nr of Images in the dataset: {len(img_meta_df)}')
img_meta_df


# In[204]:


# Visualize Image Resolutions
custom_lines = [Line2D([], [], color='red',marker='.',  markersize=5,linestyle='None'),
                Line2D([], [], color='blue', marker='.', markersize=5,linestyle='None'),
                Line2D([], [], color='green', marker='.', markersize=5,linestyle='None')]

def pltcolor(lst):
    L3_string = "L3"
    H1G_string = "H1G"
    L2_string = "L2"
    cols=[]
    for l in lst:
        if H1G_string in l:
            cols.append('red')
        elif L2_string in l:
            cols.append('blue')
        else:
            cols.append('green')
    return cols
# Create the colors list using the function above
cols=pltcolor(img_meta)

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111)
points = ax.scatter(img_meta_df.Width, img_meta_df.Height, color=cols, alpha=0.9, s=img_meta_df["Aspect Ratio"]*100, picker=True)
ax.set_title("Chip Dimensions", size=20)
ax.set_xlabel("Width", size=16)
ax.set_ylabel("Height", size=16)
ax.grid()

ax.legend(custom_lines, ['H1G', 'L2', 'L3'])
fig.savefig(r'D:\soybean_yield\figures\image_resolutions_stack_chips.png', dpi=300,bbox_inches='tight')


# In[326]:


## Split the dataset into training and testing
from sklearn.model_selection import train_test_split
X=data_shuffled['image_path'].values
y=data_shuffled['yield_kgha'].values

## Split training and testing
X_train, X_test, y_train, y_test = train_test_split(data_shuffled['image_path'], data_shuffled['yield_kgha'], test_size=0.20, random_state=0,shuffle=False)


# In[327]:


## Split training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


# In[328]:


## Print out the sizes of training and testing data
print("training data size: ",X_train.size)
print("validation data size: ",X_val.size)
print("test data size: ",X_test.size)


# In[329]:


## Print out the sizes of training and testing data
print("training data size: ",y_train.size)
print("validation data size: ",y_val.size)
print("test data size: ",y_test.size)


# In[330]:


## Create a list of .tif file
import os 
import glob
file_list_train = X_train
file_list_val = X_val
file_list_test = X_test


# In[ ]:


# Create file list for training data
resize_list=[]
for i in file_list_train:
    image=gdal.Open(i).ReadAsArray()
    #image_rescale=img_trans/255
    image_t=image.transpose(2,1,0)
    image_resized = resize(image_t, (220,600,20),anti_aliasing=False)
    resize_list.append(image_resized)

train_data=np.stack(resize_list, axis=0)
print(train_data.shape)


# In[ ]:


## Make sure the imagery looks right
plt.figure(1,(24,10)).subplots_adjust(hspace=0.5)
plt.subplot(411)
plt.title('Image 10')
plt.imshow(train_data[10,:,:,0],cmap='gray')

plt.subplot(412)
plt.title('Image 20')
plt.imshow(train_data[100,:,:,4],cmap='gray')

plt.subplot(413)
plt.title('Image 30')
plt.imshow(train_data[300,:,:,10],cmap='gray')

plt.subplot(414)
plt.title('Image 40')
plt.imshow(train_data[900,:,:,19],cmap='gray')

plt.savefig(r'D:\soybean_yield\figures\train_image_stack.png', dpi=300,bbox_inches='tight')
plt.show()


# In[ ]:


# Create file list for training data
resize_list=[]
for i in file_list_train:
    image=gdal.Open(i).ReadAsArray()
    #image_rescale=img_trans/255
    image_t=image.transpose(2,1,0)
    image_resized = resize(image_t, (220,600,20),anti_aliasing=False)
    resize_list.append(image_resized)

val_data=np.stack(resize_list, axis=0)
print(val_data.shape)


# In[ ]:


## Make sure the imagery looks right
plt.figure(1,(24,10)).subplots_adjust(hspace=0.5)
plt.subplot(411)
plt.title('Image 10')
plt.imshow(val_data[10,:,:,0],cmap='gray')

plt.subplot(412)
plt.title('Image 20')
plt.imshow(val_data[100,:,:,4],cmap='gray')

plt.subplot(413)
plt.title('Image 30')
plt.imshow(val_data[300,:,:,10],cmap='gray')

plt.subplot(414)
plt.title('Image 40')
plt.imshow(val_data[900,:,:,19],cmap='gray')

plt.savefig(r'D:\soybean_yield\figures\val_image_stack.png', dpi=300,bbox_inches='tight')
plt.show()


# In[ ]:


# Create file list for test data
resize_list=[]
for i in file_list_test:
    image=gdal.Open(i).ReadAsArray()
    #image_rescale=img_trans/255
    image_t=image.transpose(2,1,0)
    image_resized = resize(image_t, (220,600,20),anti_aliasing=False)
    resize_list.append(image_resized)

test_data=np.stack(resize_list, axis=0)
print(test_data.shape)


# In[ ]:


## Make sure the imagery looks right
plt.figure(1,(24,10)).subplots_adjust(hspace=0.5)
plt.subplot(411)
plt.title('Image 10')
plt.imshow(test_data[10,:,:,0],cmap='gray')

plt.subplot(412)
plt.title('Image 20')
plt.imshow(test_data[100,:,:,1],cmap='gray')

plt.subplot(413)
plt.title('Image 30')
plt.imshow(test_data[200,:,:,2],cmap='gray')

plt.subplot(414)
plt.title('Image 40')
plt.imshow(test_data[250,:,:,3],cmap='gray')

plt.savefig(r'D:\soybean_yield\figures\stack_test_image.png', dpi=300,bbox_inches='tight')
plt.show()


# In[ ]:


## Plot the pixel values of the training, validation, and testing data
## Flatten training and testing data first

a = train_data.flatten()
b= test_data.flatten()

plt.figure(figsize=(32, 6))
plt.subplot(1,2,1, label='Pixel Distribution')
plt.hist(a, bins=50, color= 'gray',edgecolor='black', linewidth=1.2)
plt.title('Pixel Distribution (Training)', fontsize=20);
plt.ylabel('Pixel Count', fontsize=16)
plt.xlabel('Pixel Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.subplot(1,2,2)
plt.hist(b, bins=50, color= 'gray',edgecolor='black', linewidth=1.2)
plt.title('Pixel Distribution (Testing)', fontsize=20);
plt.ylabel('Pixel Count', fontsize=16)
plt.xlabel('Pixel Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.savefig(r'D:\soybean_yield\figures\stack_pix_dist.png', dpi=300,bbox_inches='tight')
plt.show()


# In[22]:


# Print min/max values
print("training: ",np.min(train_data),np.max(train_data))
print("validation: ",np.min(val_data),np.max(val_data))
print("testing: ",np.min(test_data),np.max(test_data))


# In[23]:


#normalize all values to be between 0 and 1
train_data = (train_data-np.min(train_data))/(np.max(train_data)-np.min(train_data))
val_data = (val_data-np.min(val_data))/(np.max(val_data)-np.min(val_data))
test_data = (test_data-np.min(test_data))/(np.max(test_data)-np.min(test_data))                                           


# In[24]:


# Print min/max values
print("training: ",np.min(train_data),np.max(train_data))
print("validation ",np.min(val_data),np.max(val_data))
print("testing: ",np.min(test_data),np.max(test_data))


# In[25]:


# Print data shape
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)


# In[26]:


# Flatten the images.
# data 220*600*20=2640000
x_train_flat = train_data.reshape(-1,2640000)
x_val_flat = val_data.reshape(-1,2640000)
x_test_flat = test_data.reshape(-1,2640000)


# In[27]:


# Print flattened array shape
print(x_train_flat.shape)
print(x_val_flat.shape)
print(x_test_flat.shape)


# In[28]:


# Reset index so it matches the PCA index
y_train_df=y_train.reset_index()
y_train_df = y_train_df.drop('index', axis=1)
y_train_df


# In[29]:


# Create pixel dataframe of flattened data
import pandas as pd
feat_cols = ['pixel'+str(i) for i in range(x_train_flat.shape[1])]
df_images = pd.DataFrame(x_train_flat,columns=feat_cols)
df_images['label'] = y_train_df
print('Size of the dataframe: {}'.format(df_images.shape))


# In[30]:


# Show dataframe head
df_images.head()


# In[31]:


## Check if any NANs exist
df_images['label'].isnull().sum()


# In[32]:


# Create the PCA method and pass the number of components as two and fit data.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df_images.iloc[:,:-1])

principal_Df = pd.DataFrame(data = principalComponents,columns = ['principal component 1', 'principal component 2'])
principal_Df['y'] = y_train_df


# In[33]:


# Print DF header
principal_Df.head()


# In[34]:


# Print explained variation
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# In[35]:


# Plot PCA 1 and 2.
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="principal component 1", y="principal component 2",
    hue="y",
    #palette=sns.color_palette("hls", 10),
    data=principal_Df,
    #legend="full",
    alpha=0.3
)
plt.title('PCA Clusters', fontsize=20);
plt.savefig(r'D:\soybean_yield\figures\stack_pca_scatter.png', dpi=300,bbox_inches='tight')


# In[36]:


## Train ML models on pca data
pca = PCA(0.9)


# In[261]:


#pca= PCA(n_components=1024)


# In[262]:


# Fit data
pca.fit(x_train_flat)


# In[263]:


## Show PCA parameters
PCA(copy=True, iterated_power='auto', n_components=0.9, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)


# In[264]:


# Print number of components
pca_comp=pca.n_components_
print(pca_comp)


# In[286]:


# Transform data
train_img_pca = pca.transform(x_train_flat)
val_img_pca = pca.transform(x_val_flat)
test_img_pca = pca.transform(x_test_flat)


# In[287]:


# Flatten the images.
# data 220*600*20=2640000
#train_img_pca_2d = train_img_pca.reshape(train_img_pca.shape[0], 32, 32, 1).astype('float32')
#x_val_flat = val_data.reshape(-1,528000)
#val_img_pca_2d = val_img_pca.reshape(val_img_pca.shape[0], 32, 32, 1).astype('float32')
#test_img_pca_2d = test_img_pca.reshape(test_img_pca.shape[0], 32, 32, 1).astype('float32')


# In[288]:


# Print flattened array shape
print(train_img_pca.shape)
print(val_img_pca.shape)
print(test_img_pca.shape)


# In[41]:


# Conduct PLS regression
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score

## Hyperparameters
n_components = [2,3,4,5,6,7,8,9,10]
scale = [True,False]
param_distribs = {'n_components':n_components,
 'scale': scale}

skfold = RepeatedKFold(n_splits=10,n_repeats=10,random_state=1)
estimator = PLSRegression()
grid = RandomizedSearchCV(estimator, param_distribs, verbose=2,n_jobs=-1,cv=skfold)
print('grid: ', grid)


# In[42]:


grid.fit(train_img_pca, y_train)


# In[43]:


# Create a df from best results
pls_grid = pd.DataFrame(grid.cv_results_)
pls_grid


# In[44]:


# best model estimator
best_model_pls = grid.best_estimator_
best_model_pls


# In[45]:


# model prediction
y_pred_pls = best_model_pls.predict(train_img_pca)
y_pred_pls_test = best_model_pls.predict(test_img_pca)


# In[46]:


## get model error (residuals)
residuals_train_pls= y_train - np.squeeze(y_pred_pls)
residuals_test_pls= y_test - np.squeeze(y_pred_pls_test)


# In[47]:


# Train MSE, RMSE, R2
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

pls_train_mse=mean_squared_error(y_train, y_pred_pls)
pls_train_rmse=np.sqrt(mean_squared_error(y_train, y_pred_pls))
pls_train_r2 = r2_score(y_train, y_pred_pls)
pls_train_ground_truth_mean = np.mean(y_train)
pls_train_rrmse = 100*pls_train_rmse/pls_train_ground_truth_mean

print('MSE: ', pls_train_mse)
print('RMSE: ', pls_train_rmse)
print('R2: ', pls_train_r2)
print('RRMSE: ', pls_train_rrmse)


# In[48]:


# Test MSE, RMSE, R2
pls_test_mse=mean_squared_error(y_test, y_pred_pls_test)
pls_test_rmse=np.sqrt(mean_squared_error(y_test, y_pred_pls_test))
pls_test_r2 = r2_score(y_test, y_pred_pls_test)
pls_test_ground_truth_mean = np.mean(y_test)
pls_test_rrmse = 100*pls_test_rmse/pls_test_ground_truth_mean

print('MSE: ', pls_test_mse)
print('RMSE: ', pls_test_rmse)
print('R2: ', pls_test_r2)
print('RRMSE: ', pls_test_rrmse)


# In[49]:


## Plot yield predictions
## What's the initial results you can get.
plt.figure(figsize=(14,6))

plt.subplot(1,2,1, label='Estimated Yield')
# Plot training & validation loss values
scatter=sns.regplot(x=y_train, y=y_pred_pls, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Training)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")
text_box = AnchoredText('MSE='+str(round(pls_train_mse,3))+'\n''RMSE='+str(round(pls_train_rmse,3))+'\n''RRMSE='+str(round(pls_train_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(pls_train_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)


plt.subplot(1,2,2)
# Plot training & validation accuracy values
scatter=sns.regplot(x=y_test, y=y_pred_pls_test, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Testing)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
text_box = AnchoredText('MSE='+str(round(pls_test_mse,3))+'\n''RMSE='+str(round(pls_test_rmse,3))+'\n''RRMSE='+str(round(pls_test_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(pls_test_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)

##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\pls_stack_pca.png', dpi=300,bbox_inches='tight')
plt.show()


# In[50]:


## Plot coefficients values
a = residuals_train_pls
b= residuals_test_pls

plt.figure(figsize=(14,6))
plt.subplot(1,2,1, label='Estimated Yield')
sns.distplot(a, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(a, bins=50,color= 'lightgreen',edgecolor='black', linewidth=1.2)
# the histogram of the data
plt.title('Error Distribution (Training)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.subplot(1,2,2)
sns.distplot(b, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(b, bins=50 ,color= 'lightgreen',edgecolor='black', linewidth=1.2)
plt.title('Error Distribution (Testing)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\residual_plots_stack_pca_PLS.png', dpi=300,bbox_inches='tight')
plt.show()


# In[51]:


import joblib
joblib.dump(best_model_pls, r'D:\soybean_yield\model_results\pls_stack_model')


# In[52]:


# load the model from disk
#best_model_pls= joblib.load(r'D:\soybean_yield\model_results\pls_model')
#result = best_model_dpls.score(d2_x_test, y_test)
#print(result)


# In[53]:


# Import the decision tree classifier from sklearn
from sklearn.tree import DecisionTreeRegressor
import time
import numpy as np

skfold = RepeatedKFold(n_splits=10,n_repeats=10,random_state=1)

## Hyperparameters
criterion = ['squared_error',  'absolute_error']
max_features = ['auto', 'sqrt', 'log2']
#max_features = [0,10,50,100,200]
splitter = ['best']
max_depth = [int(x) for x in np.linspace(1, 20, num = 20)]
max_depth.append(None)
max_depth=[2,4,6]
min_samples_split = [2,4,6]
min_samples_leaf = [2,4,6]
param_distribs = {'criterion':criterion,
 'max_features': max_features,
 'max_depth': max_depth,
 'splitter': splitter,
 'min_samples_split': min_samples_split,
 'min_samples_leaf': min_samples_leaf}

estimator = DecisionTreeRegressor()
grid = RandomizedSearchCV(estimator, param_distribs, verbose=2,n_jobs=-1,cv=skfold)
start_time = time.time()
grid.fit(train_img_pca, y_train)

elapsed_time = time.time() - start_time
print(f'{elapsed_time:.2f}s elapsed during training')


# In[54]:


# Create a df from best results
dt_grid = pd.DataFrame(grid.cv_results_)
dt_grid


# In[55]:


## View best model parameters
best_model_dt = grid.best_estimator_
best_model_dt


# In[56]:


## Predict on training and testing data
y_pred_dt = best_model_dt.predict(train_img_pca)
y_pred_dt_test = best_model_dt.predict(test_img_pca)


# In[57]:


## get model error (residuals)
residuals_train_dt= y_train - np.squeeze(y_pred_dt)
residuals_test_dt= y_test - np.squeeze(y_pred_dt_test)


# In[58]:


# Train MSE, RMSE, R2
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

dt_train_mse=mean_squared_error(y_train, y_pred_dt)
dt_train_rmse=np.sqrt(mean_squared_error(y_train, y_pred_dt))
dt_train_r2 = r2_score(y_train, y_pred_dt)
dt_train_ground_truth_mean = np.mean(y_train)
dt_train_rrmse = 100*dt_train_rmse/dt_train_ground_truth_mean

print('MSE: ', dt_train_mse)
print('RMSE: ', dt_train_rmse)
print('R2: ', dt_train_r2)
print('RRMSE: ', dt_train_rrmse)


# In[59]:


# Test MSE, RMSE, R2
dt_test_mse=mean_squared_error(y_test, y_pred_dt_test)
dt_test_rmse=np.sqrt(mean_squared_error(y_test, y_pred_dt_test))
dt_test_r2 = r2_score(y_test, y_pred_dt_test)
dt_test_ground_truth_mean = np.mean(y_test)
dt_test_rrmse = 100*dt_test_rmse/dt_test_ground_truth_mean

print('MSE: ', dt_test_mse)
print('RMSE: ', dt_test_rmse)
print('R2: ', dt_test_r2)
print('RRMSE: ', dt_test_rrmse)


# In[60]:


## Plot yield predictions
## What's the initial results you can get.
plt.figure(figsize=(14,6))

plt.subplot(1,2,1, label='Estimated Yield')
# Plot training & validation loss values
scatter=sns.regplot(x=y_train, y=y_pred_dt, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Training)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")
text_box = AnchoredText('MSE='+str(round(dt_train_mse,3))+'\n''RMSE='+str(round(dt_train_rmse,3))+'\n''RRMSE='+str(round(dt_train_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(dt_train_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)


plt.subplot(1,2,2)
# Plot training & validation accuracy values
scatter=sns.regplot(x=y_test, y=y_pred_dt_test, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Testing)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
text_box = AnchoredText('MSE='+str(round(dt_test_mse,3))+'\n''RMSE='+str(round(dt_test_rmse,3))+'\n''RRMSE='+str(round(dt_test_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(dt_test_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)

##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\stack_pca_DT.png', dpi=300,bbox_inches='tight')
plt.show()


# In[61]:


## Plot coefficients values
a = residuals_train_dt
b= residuals_test_dt

plt.figure(figsize=(14,6))
plt.subplot(1,2,1, label='Estimated Yield')
sns.distplot(a, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(a, bins=50,color= 'lightgreen',edgecolor='black', linewidth=1.2)
# the histogram of the data
plt.title('Error Distribution (Training)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.subplot(1,2,2)
sns.distplot(b, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(b, bins=50 ,color= 'lightgreen',edgecolor='black', linewidth=1.2)
plt.title('Error Distribution (Testing)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\residual_plots_stack_pca_DT.png', dpi=300,bbox_inches='tight')
plt.show()


# In[62]:


import joblib
joblib.dump(best_model_dt, r'D:\soybean_yield\model_results\stack_dt_model')


# In[63]:


# load the model from disk
#best_model_dt= joblib.load(r'D:\soybean_yield\model_results\dt_model')
#result = best_model_dt.score(d2_x_test, y_test)
#print(result)


# In[64]:


# Import the random forest classifier from sklearn
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time

skfold = RepeatedKFold(n_splits=10,n_repeats=10,random_state=1)

## Hyperparameters
n_estimators=[1000]
criterion = ['squared_error','absolute_error']
max_features = ['auto', 'sqrt', 'log2']
max_depth=[2,4,6]
max_depth = [int(x) for x in np.linspace(1, 20, num = 20)]
max_depth.append(None)
min_samples_split = [2,4,6]
min_samples_leaf = [2,4,6]
param_distribs = {
 'n_estimators':n_estimators,
 'criterion':criterion,
 'max_features': max_features,
 'max_depth': max_depth,
 'min_samples_split': min_samples_split,
 'min_samples_leaf': min_samples_leaf
 }

estimator = RandomForestRegressor(verbose=10)
start_time = time.time()
grid = RandomizedSearchCV(estimator, param_distribs, verbose=1,n_jobs=-1,cv=skfold)
grid.fit(train_img_pca, y_train)
elapsed_time = time.time() - start_time
print(f'{elapsed_time:.2f}s elapsed during training')


# In[65]:


# Create a df from best results
rf_grid = pd.DataFrame(grid.cv_results_)
rf_grid


# In[66]:


## Show model parameters
best_model_rf = grid.best_estimator_
best_model_rf


# In[67]:


## Predict on training and testing data
y_pred_rf = best_model_rf.predict(train_img_pca)
y_pred_rf_test = best_model_rf.predict(test_img_pca)


# In[68]:


## get model error (residuals)
residuals_train_rf= y_train - np.squeeze(y_pred_rf)
residuals_test_rf= y_test - np.squeeze(y_pred_rf_test)


# In[69]:


# Train MSE, RMSE, R2
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

rf_train_mse=mean_squared_error(y_train, y_pred_rf)
rf_train_rmse=np.sqrt(mean_squared_error(y_train, y_pred_rf))
rf_train_r2 = r2_score(y_train, y_pred_rf)
rf_train_ground_truth_mean = np.mean(y_train)
rf_train_rrmse = 100*rf_train_rmse/rf_train_ground_truth_mean

print('MSE: ', rf_train_mse)
print('RMSE: ', rf_train_rmse)
print('R2: ', rf_train_r2)
print('RRMSE: ', rf_train_rrmse)


# In[70]:


# Test MSE, RMSE, R2
rf_test_mse=mean_squared_error(y_test, y_pred_rf_test)
rf_test_rmse=np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
rf_test_r2 = r2_score(y_test, y_pred_rf_test)
rf_test_ground_truth_mean = np.mean(y_test)
rf_test_rrmse = 100*rf_test_rmse/rf_test_ground_truth_mean

print('MSE: ', rf_test_mse)
print('RMSE: ', rf_test_rmse)
print('R2: ', rf_test_r2)
print('RRMSE: ', rf_test_rrmse)


# In[71]:


## Plot yield predictions
## What's the initial results you can get.
plt.figure(figsize=(14,6))

plt.subplot(1,2,1, label='Estimated Yield')
# Plot training & validation loss values
scatter=sns.regplot(x=y_train, y=y_pred_rf, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Training)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")
text_box = AnchoredText('MSE='+str(round(rf_train_mse,3))+'\n''RMSE='+str(round(rf_train_rmse,3))+'\n''RRMSE='+str(round(rf_train_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(rf_train_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)


plt.subplot(1,2,2)
# Plot training & validation accuracy values
scatter=sns.regplot(x=y_test, y=y_pred_rf_test, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Testing)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
text_box = AnchoredText('MSE='+str(round(rf_test_mse,3))+'\n''RMSE='+str(round(rf_test_rmse,3))+'\n''RRMSE='+str(round(rf_test_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(rf_test_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)

##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\rf_stack_pca.png', dpi=300,bbox_inches='tight')
plt.show()


# In[72]:


## Plot coefficients values
a = residuals_train_rf
b= residuals_test_rf

plt.figure(figsize=(14,6))
plt.subplot(1,2,1, label='Estimated Yield')
sns.distplot(a, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(a, bins=50,color= 'lightgreen',edgecolor='black', linewidth=1.2)
# the histogram of the data
plt.title('Error Distribution (Training)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.subplot(1,2,2)
sns.distplot(b, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(b, bins=50 ,color= 'lightgreen',edgecolor='black', linewidth=1.2)
plt.title('Error Distribution (Testing)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\stack_residual_plots_pca_RF.png', dpi=300,bbox_inches='tight')
plt.show()


# In[73]:


import joblib
joblib.dump(best_model_rf, r'D:\soybean_yield\model_results\rf_model')


# In[74]:


# load the model from disk
#best_model_rf= joblib.load(r'D:\soybean_yield\model_results\rf_model')
#result = best_model_rf.score(d2_x_test, y_test)
#print(result)


# In[75]:


## Hyperparameters
from sklearn.svm import SVR
kernel = ['linear', 'poly', 'rbf']
degree = [2, 3, 4, 5]
gamma = ['auto']
param_distribs = {'kernel':kernel,
 'degree': degree,
 'gamma': gamma}

estimator = SVR()
grid = RandomizedSearchCV(estimator, param_distribs, verbose=1,n_jobs=-1,cv=skfold)
grid.fit(train_img_pca, y_train)


# In[76]:


# Create a df from best results
svr_grid = pd.DataFrame(grid.cv_results_)
svr_grid


# In[77]:


## View best model parameters
best_model_sv = grid.best_estimator_
best_model_sv


# In[78]:


## Predict on training and testing data
y_pred_sv = best_model_sv.predict(train_img_pca)
y_pred_sv_test = best_model_sv.predict(test_img_pca)


# In[79]:


## get model error (residuals)
residuals_train_sv= y_train - np.squeeze(y_pred_sv)
residuals_test_sv= y_test - np.squeeze(y_pred_sv_test)


# In[80]:


# Train MSE, RMSE, R2
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

sv_train_mse=mean_squared_error(y_train, y_pred_sv)
sv_train_rmse=np.sqrt(mean_squared_error(y_train, y_pred_sv))
sv_train_r2 = r2_score(y_train, y_pred_sv)
sv_train_ground_truth_mean = np.mean(y_train)
sv_train_rrmse = 100*sv_train_rmse/sv_train_ground_truth_mean

print('MSE: ', sv_train_mse)
print('RMSE: ', sv_train_rmse)
print('R2: ', sv_train_r2)
print('RRMSE: ', sv_train_rrmse)


# In[81]:


# Test MSE, RMSE, R2
sv_test_mse=mean_squared_error(y_test, y_pred_sv_test)
sv_test_rmse=np.sqrt(mean_squared_error(y_test, y_pred_sv_test))
sv_test_r2 = r2_score(y_test, y_pred_sv_test)
sv_test_ground_truth_mean = np.mean(y_test)
sv_test_rrmse = 100*sv_test_rmse/sv_test_ground_truth_mean

print('MSE: ', sv_test_mse)
print('RMSE: ', sv_test_rmse)
print('R2: ', sv_test_r2)
print('RRMSE: ', sv_test_rrmse)


# In[82]:


## Plot yield predictions
## What's the initial results you can get.
plt.figure(figsize=(14,6))

plt.subplot(1,2,1, label='Estimated Yield')
# Plot training & validation loss values
scatter=sns.regplot(x=y_train, y=y_pred_sv, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Training)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")
text_box = AnchoredText('MSE='+str(round(sv_train_mse,3))+'\n''RMSE='+str(round(sv_train_rmse,3))+'\n''RRMSE='+str(round(sv_train_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(sv_train_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)


plt.subplot(1,2,2)
# Plot training & validation accuracy values
scatter=sns.regplot(x=y_test, y=y_pred_sv_test, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Testing)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
text_box = AnchoredText('MSE='+str(round(sv_test_mse,3))+'\n''RMSE='+str(round(sv_test_rmse,3))+'\n''RRMSE='+str(round(sv_test_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(sv_test_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)

##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\sv_stack_pca.png', dpi=300,bbox_inches='tight')
plt.show()


# In[83]:


## Plot coefficients values
a = residuals_train_sv
b= residuals_test_sv

plt.figure(figsize=(14,6))
plt.subplot(1,2,1, label='Estimated Yield')
sns.distplot(a, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(a, bins=50,color= 'lightgreen',edgecolor='black', linewidth=1.2)
# the histogram of the data
plt.title('Error Distribution (Training)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.subplot(1,2,2)
sns.distplot(b, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(b, bins=50 ,color= 'lightgreen',edgecolor='black', linewidth=1.2)
plt.title('Error Distribution (Testing)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\stack_residual_plots_pca_SV.png', dpi=300,bbox_inches='tight')
plt.show()


# In[84]:


import joblib
joblib.dump(best_model_sv, r'D:\soybean_yield\model_results\sv_model')


# In[85]:


# load the model from disk
#best_model_sv= joblib.load(r'D:\soybean_yield\model_results\sv_model')
#result = best_model_sv.score(d2_x_test, y_test)
#print(result)


# In[86]:


# define cnn model for random search 
def create_model(neurons=10,optimizer= 'adam',init_mode='uniform',activation='relu',dropout_rate=0.0,epochs=10,batch_size=10,learn_rate=0.001):
    model = Sequential()
    model.add(Dense(neurons, kernel_initializer=init_mode,activation=activation, input_shape=(pca_comp,)))
    model.add(Dense(neurons, kernel_initializer=init_mode,activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, kernel_initializer=init_mode,activation=activation))
    model.add(Dense(neurons, kernel_initializer=init_mode,activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, kernel_initializer=init_mode,activation=activation))
    model.add(Dense(1, kernel_initializer=init_mode,activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[keras.metrics.RootMeanSquaredError()])
    return model


# In[87]:


# Model summary
create_model().summary()


# In[88]:


# create model
from keras.wrappers.scikit_learn import KerasRegressor
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
## Model tuning using grid search and cross validation
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skopt import BayesSearchCV

# define the grid search parameters
hyperparameter_set = {'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
                     'init_mode':['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
                     'activation':['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'],
                      'dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                      'neurons':[10,20,30,40,50,100,200,400,800,1000],
                      'epochs':[10, 50, 100,200,500],
                      'batch_size':[10, 20, 40, 60, 80, 100],
                      'learn_rate':[0.001, 0.01, 0.1, 0.2, 0.3]}


nn_layer =  KerasRegressor(build_fn=create_model, verbose=1)
nn_layer._estimator_type = "regression"
skfold = RepeatedKFold(n_splits=10,n_repeats=10,random_state=42)
grid = RandomizedSearchCV(nn_layer, hyperparameter_set, verbose=1,n_jobs=-1,cv=skfold, random_state=42)
#grid=BayesSearchCV(nn_layer, hyperparameter_set, verbose=1,n_jobs=-1,cv=skfold, random_state=42)
grid.fit(train_img_pca,y_train)


# In[89]:


# Create a df from best results
nn_grid = pd.DataFrame(grid.cv_results_)
nn_grid


# In[90]:


# Print best model parameters
print("Best: %f using %s" % (grid.best_score_, grid.best_params_))


# In[91]:


## View best model parameters
best_model_nn = grid.best_estimator_
best_model_nn


# In[92]:


## Predict on training and testing data
y_pred_nn = best_model_nn.predict(train_img_pca)
y_pred_nn_test = best_model_nn.predict(test_img_pca)


# In[93]:


## get model error (residuals)
residuals_train_nn= y_train - np.squeeze(y_pred_nn)
residuals_test_nn= y_test - np.squeeze(y_pred_nn_test)


# In[94]:


# Train MSE, RMSE, R2
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

nn_train_mse=mean_squared_error(y_train, y_pred_nn)
nn_train_rmse=np.sqrt(mean_squared_error(y_train, y_pred_nn))
nn_train_r2 = r2_score(y_train, y_pred_nn)
nn_train_ground_truth_mean = np.mean(y_train)
nn_train_rrmse = 100*nn_train_rmse/nn_train_ground_truth_mean

print('MSE: ', nn_train_mse)
print('RMSE: ', nn_train_rmse)
print('R2: ', nn_train_r2)
print('RRMSE: ', nn_train_rrmse)


# In[95]:


# Test MSE, RMSE, R2
nn_test_mse=mean_squared_error(y_test, y_pred_nn_test)
nn_test_rmse=np.sqrt(mean_squared_error(y_test, y_pred_nn_test))
nn_test_r2 = r2_score(y_test, y_pred_nn_test)
nn_test_ground_truth_mean = np.mean(y_test)
nn_test_rrmse = 100*nn_test_rmse/nn_test_ground_truth_mean

print('MSE: ', nn_test_mse)
print('RMSE: ', nn_test_rmse)
print('R2: ', nn_test_r2)
print('RRMSE: ', nn_test_rrmse)


# In[96]:


## Plot yield predictions
## What's the initial results you can get.
plt.figure(figsize=(14,6))

plt.subplot(1,2,1, label='Estimated Yield')
# Plot training & validation loss values
scatter=sns.regplot(x=y_train, y=y_pred_nn, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Training)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")
text_box = AnchoredText('MSE='+str(round(nn_train_mse,3))+'\n''RMSE='+str(round(nn_train_rmse,3))+'\n''RRMSE='+str(round(nn_train_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(nn_train_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)


plt.subplot(1,2,2)
# Plot training & validation accuracy values
scatter=sns.regplot(x=y_test, y=y_pred_nn_test, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Testing)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
text_box = AnchoredText('MSE='+str(round(nn_test_mse,3))+'\n''RMSE='+str(round(nn_test_rmse,3))+'\n''RRMSE='+str(round(nn_test_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(nn_test_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)

##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\nn1_model_stack_pca.png', dpi=300,bbox_inches='tight')
plt.show()


# In[97]:


## Plot coefficients values
a = residuals_train_nn
b= residuals_test_nn

plt.figure(figsize=(14,6))
plt.subplot(1,2,1, label='Estimated Yield')
sns.distplot(a, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(a, bins=50,color= 'lightgreen',edgecolor='black', linewidth=1.2)
# the histogram of the data
plt.title('Error Distribution (Training)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.subplot(1,2,2)
sns.distplot(b, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(b, bins=50 ,color= 'lightgreen',edgecolor='black', linewidth=1.2)
plt.title('Error Distribution (Testing)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\stack_residual_plots_pca.png', dpi=300,bbox_inches='tight')
plt.show()


# In[98]:


# define cnn model for test harness
def define_model(): 
    model = Sequential()
    model.add(Dense(100, kernel_initializer='he_uniform',activation='relu', input_shape=(pca_comp,)))
    model.add(Dense(200, kernel_initializer='he_uniform',activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(300, kernel_initializer='he_uniform',activation='relu'))
    model.add(Dense(400, kernel_initializer='he_uniform',activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(500, kernel_initializer='he_uniform',activation='relu'))
    model.add(Dense(1, kernel_initializer='he_uniform',activation='linear'))
    opt = Adam(learning_rate=0.00001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[keras.metrics.RootMeanSquaredError()])
    return model 


# In[99]:


# Print model summary
define_model().summary()


# In[100]:


## Use learning rate decay to improve model performance
batch_size = 64
epochs=500
initial_learning_rate = 0.000001
decay = initial_learning_rate / epochs

def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)

## Define model callbacks
keras_callbacks   = [LearningRateScheduler(lr_time_based_decay, verbose=1)]


# In[101]:


# Model function
def evaluate_model(dataX, dataY, n_folds=5): 
    scores, histories = list(), list()  
    kfold = KFold(n_folds, shuffle=True, random_state=1) 
    # enumerate splits 
    for train_ix, test_ix in kfold.split(dataX): 
        # define model 
        model = define_model() 
        # select rows for train and test 
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix],dataY[test_ix] 
        # fit model 
        history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_data=(testX,testY), verbose=1, callbacks=keras_callbacks) 
        # evaluate model 
        _, mse = model.evaluate(testX, testY, verbose=0) 
        #print('> %.3f' % (acc * 100.0))
        print('mean MSE=', mean(mse))
        # stores scores 
        scores.append(mse) 
        histories.append(history) 
    return scores, histories, model


# In[102]:


from matplotlib.lines import Line2D
from cycler import cycler
def summarize_diagnostics(histories): 
  plt.figure(figsize=(14,6))
  plt.rc('axes', prop_cycle=(cycler('color', ['dodgerblue', 'orange', 'green','red','purple'])))
  for i in range(len(histories)): 
    # plot loss 
    plt.subplot(1,2,1)
    # Plot training & validation loss values
    plt.plot(histories[i].history['loss'], label='_nolegend_',alpha=0.55)
    plt.plot(histories[i].history['val_loss'],label='_nolegend_',linestyle='--', alpha=0.55)
    plt.plot(mean_val_loss,label='_nolegend_',linewidth=3.0,linestyle='-.')
    plt.title('Training & Validation Loss', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.xticks( fontsize=12)
    plt.yticks( fontsize=12)
    plt.grid(True)
    text_box = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,'fontfamily':'DejaVu Sans'})
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    plt.gca().add_artist(text_box)
    custom_lines = [Line2D([0], [0], color='black', lw=1),
                    Line2D([0], [0], color='black', lw=1,linestyle='--'),
                    Line2D([0], [0], color='black', lw=1,linestyle='-.'),
                    Line2D([0], [0], color='dodgerblue', lw=1),
                    Line2D([0], [0], color='orange', lw=1),
                    Line2D([0], [0], color='green', lw=1),
                    Line2D([0], [0], color='red', lw=1),
                    Line2D([0], [0], color='purple', lw=1)
                   ]
    plt.legend(custom_lines, ['Train','Validation','Mean Validation Loss','K-fold 1', 'K-fold 2', 'K-fold 3','K-fold 4','K-fold 5'],loc='center right', fontsize=12)
  
    # plot RMSE 
    plt.subplot(1,2,2)
    # Plot training & validation accuracy values
    plt.plot(histories[i].history['root_mean_squared_error'], label='_nolegend_',alpha=0.55)
    plt.plot(histories[i].history['val_root_mean_squared_error'],label='_nolegend_',linestyle='--', alpha=0.55)
    plt.plot(mean_val_rmse,label='_nolegend_',linewidth=3.0,linestyle='-.')
    plt.title('Training & Validation RMSE', fontsize=15)
    plt.ylabel('RMSE', fontsize=15)
    plt.xlabel('Epoch', fontsize=15)
    plt.xticks( fontsize=12)
    plt.yticks( fontsize=12)
    plt.grid(True)
    text_box = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,'fontfamily':'DejaVu Sans'})
    plt.setp(text_box.patch, facecolor='white', alpha=0.5)
    plt.gca().add_artist(text_box)
    custom_lines = [Line2D([0], [0], color='black', lw=1),
                    Line2D([0], [0], color='black', lw=1,linestyle='--'),
                    Line2D([0], [0], color='black', lw=1,linestyle='-.'),
                    Line2D([0], [0], color='dodgerblue', lw=1),
                    Line2D([0], [0], color='orange', lw=1),
                    Line2D([0], [0], color='green', lw=1),
                    Line2D([0], [0], color='red', lw=1),
                    Line2D([0], [0], color='purple', lw=1)]
    plt.legend(custom_lines, ['Train','Validation','Mean Validation RMSE','K-fold 1', 'K-fold 2', 'K-fold 3','K-fold 4','K-fold 5'],loc='center right', fontsize=12)

  plt.tight_layout()
  plt.savefig(r'D:\soybean_yield\figures\stack_nn2_pca_loss_graphs_pca.png', dpi=300,bbox_inches='tight')
  plt.show()


# In[103]:


# summarize model performance 
def summarize_performance(scores): 
    # print summary 
    print('RMSE: mean=%.3f std=%.3f, n=%d' % (mean(scores), std(scores),len(scores))) 
    # box and whisker plots of results 
    plt.boxplot(scores,notch=True, patch_artist=True,meanline=True, showmeans=True) 
    plt.title('Cross Validation Comparison')
    plt.xlabel('model')
    plt.ylabel('R$^2$ Score')
    plt.savefig(r'D:\soybean_yield\figures\stack_nn2_box_pca.png', dpi=300,bbox_inches='tight')
    plt.show() 


# In[104]:


## Run network
scores, histories, model = evaluate_model(np.array(train_img_pca), y_train.values, n_folds=5) 


# In[105]:


## Mean history values
## Training loss
train_loss1=histories[0].history['loss']
train_loss2=histories[1].history['loss']
train_loss3=histories[2].history['loss']
train_loss4=histories[3].history['loss']
train_loss5=histories[4].history['loss']
multiple_lists = [train_loss1, train_loss2, train_loss3, train_loss4, train_loss5]
arrays = [np.array(x) for x in multiple_lists]
[np.mean(k) for k in zip(*arrays)]
mean_train_loss=[np.mean(k) for k in zip(*arrays)]

## Training RMSE
train_rmse1=histories[0].history['root_mean_squared_error']
train_rmse2=histories[1].history['root_mean_squared_error']
train_rmse3=histories[2].history['root_mean_squared_error']
train_rmse4=histories[3].history['root_mean_squared_error']
train_rmse5=histories[4].history['root_mean_squared_error']
multiple_lists = [train_rmse1, train_rmse2, train_rmse3, train_rmse4, train_rmse5]
arrays = [np.array(x) for x in multiple_lists]
[np.mean(k) for k in zip(*arrays)]
mean_train_rmse=[np.mean(k) for k in zip(*arrays)]

## Validation Loss
val_loss1=histories[0].history['val_loss']
val_loss2=histories[1].history['val_loss']
val_loss3=histories[2].history['val_loss']
val_loss4=histories[3].history['val_loss']
val_loss5=histories[4].history['val_loss']
multiple_lists = [val_loss1, val_loss2, val_loss3, val_loss4, val_loss5]
arrays = [np.array(x) for x in multiple_lists]
[np.mean(k) for k in zip(*arrays)]
mean_val_loss=[np.mean(k) for k in zip(*arrays)]

## Validation RMSE
val_rmse1=histories[0].history['val_root_mean_squared_error']
val_rmse2=histories[1].history['val_root_mean_squared_error']
val_rmse3=histories[2].history['val_root_mean_squared_error']
val_rmse4=histories[3].history['val_root_mean_squared_error']
val_rmse5=histories[4].history['val_root_mean_squared_error']
multiple_lists = [val_rmse1, val_rmse2, val_rmse3, val_rmse4, val_rmse5]
arrays = [np.array(x) for x in multiple_lists]
[np.mean(k) for k in zip(*arrays)]
mean_val_rmse=[np.mean(k) for k in zip(*arrays)]


# In[106]:


# View graphs
summarize_diagnostics(histories) 


# In[107]:


## Boxplot the model results
summarize_performance(scores)


# In[108]:


## Change model name for ease of processing
best_model_nn2=model


# In[109]:


## Get training and testing loss and accuracy
nn2_train_loss, nn2_train_acc = best_model_nn2.evaluate(train_img_pca,y_train)
nn2_test_loss, nn2_test_acc = best_model_nn2.evaluate(test_img_pca,y_test)

print('Training Accuracy: ', nn2_train_acc)
print('Training Loss: ', nn2_train_loss)
print('Testing Accuracy: ', nn2_test_acc)
print('Testing Loss: ', nn2_test_loss)


# In[110]:


## Predict on train and test data
y_pred_nn2_train = best_model_nn2.predict(train_img_pca)
y_pred_nn2_test = best_model_nn2.predict(test_img_pca)


# In[111]:


## get model error (residuals)
residuals_train_nn2= y_train - np.squeeze(y_pred_nn2_train)
residuals_test_nn2= y_test - np.squeeze(y_pred_nn2_test)


# In[112]:


# Train MSE, RMSE, R2
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

nn2_train_mse=mean_squared_error(y_train, y_pred_nn2_train)
nn2_train_rmse=np.sqrt(mean_squared_error(y_train, y_pred_nn2_train))
nn2_train_r2 = r2_score(y_train, y_pred_nn2_train)
nn2_train_ground_truth_mean = np.mean(y_train)
nn2_train_rrmse = 100*nn2_train_rmse/nn2_train_ground_truth_mean

print('MSE: ', nn2_train_mse)
print('RMSE: ', nn2_train_rmse)
print('R2: ', nn2_train_r2)
print('RRMSE: ', nn2_train_rrmse)


# In[113]:


# Test MSE, RMSE, R2
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

nn2_test_mse=mean_squared_error(y_test, y_pred_nn2_test)
nn2_test_rmse=np.sqrt(mean_squared_error(y_test, y_pred_nn2_test))
nn2_test_r2 = r2_score(y_test, y_pred_nn2_test)
nn2_test_ground_truth_mean = np.mean(y_test)
nn2_test_rrmse = 100*nn2_test_rmse/nn2_test_ground_truth_mean

print('MSE: ', nn2_test_mse)
print('RMSE: ', nn2_test_rmse)
print('R2: ', nn2_test_r2)
print('RRMSE: ', nn2_test_rrmse)


# In[114]:


## Plot yield predictions
## What's the initial results you can get.
plt.figure(figsize=(14,6))

plt.subplot(1,2,1, label='Estimated Yield')
# Plot training & validation loss values
scatter=sns.regplot(x=y_train, y=y_pred_nn2_train, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Training)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")
text_box = AnchoredText('MSE='+str(round(nn2_train_mse,3))+'\n''RMSE='+str(round(nn2_train_rmse,3))+'\n''RRMSE='+str(round(nn2_train_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(nn2_train_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)


plt.subplot(1,2,2)
# Plot training & validation accuracy values
scatter=sns.regplot(x=y_test, y=y_pred_nn2_test, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Testing)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
text_box = AnchoredText('MSE='+str(round(nn2_test_mse,3))+'\n''RMSE='+str(round(nn2_test_rmse,3))+'\n''RRMSE='+str(round(nn2_test_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(nn2_test_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)

##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\nn2_model_regression_stack_pca.png', dpi=300,bbox_inches='tight')
plt.show()


# In[115]:


## Plot coefficients values
a = residuals_train_nn2
b= residuals_test_nn2

plt.figure(figsize=(14,6))
plt.subplot(1,2,1, label='Estimated Yield')
sns.distplot(a, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(a, bins=50,color= 'lightgreen',edgecolor='black', linewidth=1.2)
# the histogram of the data
plt.title('Error Distribution (Training)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.subplot(1,2,2)
sns.distplot(b, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(b, bins=50 ,color= 'lightgreen',edgecolor='black', linewidth=1.2)
plt.title('Error Distribution (Testing)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\nn2_stack_residual_plots_pca.png', dpi=300,bbox_inches='tight')
plt.show()


# In[205]:


## Split the dataset into training and testing
from sklearn.model_selection import train_test_split
X=data_shuffled['image_path'].values
y=data_shuffled['yield_kgha'].values

## Split training and testing
X_train, X_test, y_train, y_test = train_test_split(data_shuffled['image_path'], data_shuffled['yield_kgha'], test_size=0.20, random_state=0,shuffle=False)


# In[206]:


## Split training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


# In[207]:


## Create a list of .tif file
import os 
import glob
file_list_train = X_train
file_list_val = X_val
file_list_test = X_test


# In[208]:


# Create file list for training data
os.chdir(r'D:/soybean_yield/macro_corpus/training_corpus_stacked/')
resize_list=[]
for i in file_list_train:
    image=gdal.Open(i).ReadAsArray()
    #image_rescale=img_trans/255
    image_t=image.transpose(2,1,0)
    image_resized = resize(image_t, (220,600,20),anti_aliasing=False)
    resize_list.append(image_resized)


# In[209]:


train_data=np.stack(resize_list, axis=0)
print(train_data.shape)


# In[210]:


# Create file list for test data
resize_list=[]
for i in file_list_val:
    image=gdal.Open(i).ReadAsArray()
    #image_rescale=img_trans/255
    image_t=image.transpose(2,1,0)
    image_resized = resize(image_t, (220,600,20),anti_aliasing=False)
    resize_list.append(image_resized)


# In[211]:


val_data=np.stack(resize_list, axis=0)
print(val_data.shape)


# In[212]:


# Create file list for test data
resize_list=[]
for i in file_list_test:
    image=gdal.Open(i).ReadAsArray()
    #image_rescale=img_trans/255
    image_t=image.transpose(2,1,0)
    image_resized = resize(image_t, (220,600,20),anti_aliasing=False)
    resize_list.append(image_resized)


# In[213]:


test_data=np.stack(resize_list, axis=0)
print(test_data.shape)


# In[214]:


# Print min/max values
print("training: ",np.min(train_data),np.max(train_data))
print("validation: ",np.min(val_data),np.max(val_data))
print("testing: ",np.min(test_data),np.max(test_data))


# In[215]:


#normalize all values to be between 0 and 1
train_data = (train_data-np.min(train_data))/(np.max(train_data)-np.min(train_data))
val_data = (val_data-np.min(val_data))/(np.max(val_data)-np.min(val_data)) 
test_data = (test_data-np.min(test_data))/(np.max(test_data)-np.min(test_data)) 


# In[216]:


# Print min/max values
print("training: ",np.min(train_data),np.max(train_data))
print("validation: ",np.min(val_data),np.max(val_data))
print("testing: ",np.min(test_data),np.max(test_data))


# In[217]:


# Print data shape
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)


# In[218]:


# Flatten the images.
# data 220*600*20=2640000
x_train_flat = train_data.reshape(-1,2640000)
x_val_flat = val_data.reshape(-1,2640000)
x_test_flat = test_data.reshape(-1,2640000)


# In[219]:


# Print flattened array shape
print(x_train_flat.shape)
print(x_val_flat.shape)
print(x_test_flat.shape)


# In[220]:


from sklearn.decomposition import PCA
#pca= PCA(n_components=1024)
pca = PCA(0.9)


# In[221]:


# Fit data
pca.fit(x_train_flat)


# In[222]:


# Print number of components
pca_comp=pca.n_components_
print(pca_comp)


# In[223]:


# Transform data
train_img_pca = pca.transform(x_train_flat)
val_img_pca = pca.transform(x_val_flat)
test_img_pca = pca.transform(x_test_flat)


# In[260]:


# Flatten the images.
# data 220*600*20=2640000
train_img_pca_2d = train_img_pca.reshape(train_img_pca.shape[0], 32, 32, 1).astype('float32')
val_img_pca_2d = val_img_pca.reshape(val_img_pca.shape[0], 32, 32, 1).astype('float32')
test_img_pca_2d = test_img_pca.reshape(test_img_pca.shape[0], 32, 32, 1).astype('float32')


# In[225]:


# Print flattened array shape
#print(train_img_pca_2d.shape)
#print(val_img_pca_2d .shape)
#print(test_img_pca_2d.shape)

print(train_img_pca.shape)
print(val_img_pca .shape)
print(test_img_pca.shape)


# In[226]:


d1_x_train = train_img_pca.reshape(1035, 189,1)
d1_x_val = val_img_pca.reshape(116, 189,1)
d1_x_test = test_img_pca.reshape(288, 189,1)


# In[227]:


import tensorflow 
import pandas as pd
import numpy as np
import os
import keras
import random
import cv2
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Convolution2D,BatchNormalization
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore")
from keras.layers import Conv1D
from tensorflow.keras.layers import Flatten,MaxPooling1D,Dropout,AvgPool1D,GlobalAveragePooling1D,ZeroPadding1D
from tensorflow.keras.layers import ReLU, concatenate


# In[228]:


print("Tensorflow-version:", tensorflow.__version__)


# In[229]:


## 1D
def conv_layer(x, filters, kernel=1, strides=1):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv1D(filters, kernel, strides=strides)(x)
    x = ZeroPadding1D(padding=(0,1))(x)
    return x
def dense_block(x, repetition, filters):
    for _ in range(repetition):
        y = conv_layer(x, 4 * filters)
        y = conv_layer(y, filters, 3)
        x = concatenate([y, x])
        return x
def transition_layer(x):
    x = conv_layer(x, x.shape[-1]/ 2)
    x = AvgPool1D(2, strides=2)(x)
    x = ZeroPadding1D(padding=(0,1))(x)
    return x
def densenet(input_shape, n_classes, filters=32):
    input = Input(input_shape)
    x = Conv1D(64, 7, strides=2)(input)
    x = ZeroPadding1D(padding=(0,1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling1D(3, strides=2)(x)
    x = ZeroPadding1D(padding=1)(x)
    for repetition in [6, 12, 24, 16]:
        d = dense_block(x, repetition,filters)
        x = transition_layer(d)
        x = GlobalAveragePooling1D()(d)
        output = Dense(n_classes, activation="linear")(x)
        model = Model(input, output)
        return model


# In[231]:


input_shape = (189,1)
n_classes = 1

model = densenet(input_shape, n_classes, filters=32)
model.summary()


# In[232]:


# compile model 
opt = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=[keras.metrics.RootMeanSquaredError()])


# In[233]:


## Use learning rate decay to improve model performance
epochs=500
batch_size= 128
initial_learning_rate = 0.001
decay = initial_learning_rate / epochs

def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)


# In[234]:


## Define model callbacks
keras_callbacks   = [
      EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.0001),
      ModelCheckpoint('D:/soybean_yield/callbacks/checkmodel.model_1', monitor='val_loss', save_best_only=True, mode='min'),
                      LearningRateScheduler(lr_time_based_decay, verbose=1)
]


# In[235]:


history = model.fit(d1_x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(d1_x_val, y_val), verbose=1, callbacks=keras_callbacks) 


# In[268]:


first_model = DenseNet121(weights = None, input_shape=(32, 32, 1), include_top=False)


# In[270]:


x=first_model.output
x= GlobalAveragePooling2D()(x)
x= BatchNormalization()(x)
x= Dropout(0.5)(x)
x= Dense(1024,activation='swish')(x) 
x= Dense(512,activation='swish')(x) 
x= BatchNormalization()(x)
x= Dropout(0.5)(x)
x= Flatten()(x)
preds=Dense(1,activation='linear')(x) #FC-layer


# In[271]:


model=Model(inputs=first_model.input,outputs=preds)
model.summary()


# In[238]:


for layer in model.layers[:-9]:
    layer.trainable=False
    
for layer in model.layers[-9:]:
    layer.trainable=True


# In[272]:


# compile model 
opt = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=[keras.metrics.RootMeanSquaredError()])


# In[273]:


## Use learning rate decay to improve model performance
epochs=500
batch_size= 128
initial_learning_rate = 0.001
decay = initial_learning_rate / epochs

def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)


# In[289]:


## Define model callbacks
keras_callbacks   = [
      EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.0001),
    LearningRateScheduler(lr_time_based_decay, verbose=1)
      #ModelCheckpoint('D:/soybean_yield/callbacks/checkmodel.model_1', monitor='val_loss', save_best_only=True, mode='min'),
                      #LearningRateScheduler(lr_time_based_decay, verbose=1)
]


# In[290]:


datagen = ImageDataGenerator()


# In[ ]:


datagen_test = ImageDataGenerator()


# In[244]:


#train_generator= datagen.flow(train_data, y_train)
#val_generator= datagen.flow(val_data, y_val)
#test_generator= datagen.flow(test_data, y_test)


# In[291]:


train_generator= datagen.flow(train_img_pca_2d, y_train)
val_generator= datagen.flow(val_img_pca_2d, y_val)
test_generator= datagen.flow(test_img_pca_2d, y_test)


# In[246]:


#anne = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-3)


# In[247]:


#history = model.fit(train_img_pca_2d, y_train, epochs=epochs, batch_size=batch_size, validation_data=(val_img_pca_2d, y_val), verbose=1, callbacks=keras_callbacks) 


# In[292]:


history=model.fit(train_generator,
                    validation_data=(val_generator),
                    steps_per_epoch = train_generator.n//train_generator.batch_size,
                    validation_steps = val_generator.n//val_generator.batch_size,
                    epochs=epochs,callbacks=keras_callbacks)


# In[293]:


# history = model.fit(train_data, y_train, epochs=epochs, batch_size=batch_size, validation_data=(val_data, y_val), verbose=1, callbacks=keras_callbacks) 


# In[294]:


#anne = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-3)
#checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)


## Fits-the-model
#history = model.fit_generator(datagen.flow(train_img_pca_2d, y_train, batch_size=128),
#               steps_per_epoch=train_img_pca_2d.shape[0] //128,
#               epochs=100,
#               verbose=2,
#               callbacks=[keras_callbacks, checkpoint],
#               validation_data=(train_img_pca_2d, y_train))


# In[295]:


# save model and architecture to single file
#model_1.save(r'D:\soybean_yield\model_results\adrian_csci4760_project5_model_1.h5')


# In[296]:


# load the model from disk
#from tensorflow.keras.models import load_model
#model_1 = load_model(r'D:\soybean_yield\model_results\adrian_csci4760_project5_model_1.h5')
#result_1 = model_1.evaluate(x_test, y_test)
#print(result_1)


# In[297]:


## Plot training loss
import matplotlib.pyplot as plt
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Training & Validation Loss', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True)
text_box = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)


plt.subplot(1,2,2)
# Plot training & validation accuracy values
plt.plot(history.history['root_mean_squared_error'], label='Train')
plt.plot(history.history['val_root_mean_squared_error'], label='Validation')
plt.title('Training & Validation RMSE', fontsize=15)
plt.ylabel('Epoch', fontsize=15)
plt.xlabel('RMSE', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True)
text_box = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.tight_layout()
#plt.savefig(r'D:\soybean_yield\figures\adrian_csci4760_project5_model_1_loss_graphs.png', dpi=300)
plt.show()


# In[298]:


## Get training and testing loss and accuracy
train_loss_1, train_acc_1 = model.evaluate(train_generator)
test_loss_1, test_acc_1 = model.evaluate(test_generator)

#train_loss_1, train_acc_1 = model.evaluate(train_img_pca_2d,y_train)
#test_loss_1, test_acc_1 = model.evaluate(test_img_pca_2d, y_test)

#train_loss_1, train_acc_1 = model.evaluate(d1_x_train,y_train)
#test_loss_1, test_acc_1 = model.evaluate(d1_x_test, y_test)

print('Training Accuracy: ', train_acc_1)
print('Training Loss: ', train_loss_1)
print('Testing Accuracy: ', test_acc_1)
print('Testing Loss: ', test_loss_1)


# In[305]:


## Predict on train and test data
y_pred_dense_train = model.predict(train_img_pca_2d)
y_pred_dense_test = model.predict(test_img_pca_2d)


# In[306]:


## get model error (residuals)
residuals_dense_train= y_train - np.squeeze(y_pred_dense_train)
residuals_dense_test= y_test - np.squeeze(y_pred_dense_test)


# In[307]:


# Train MSE, RMSE, R2
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

dense_train_mse=mean_squared_error(y_train, y_pred_dense_train)
dense_train_rmse=np.sqrt(mean_squared_error(y_train, y_pred_dense_train))
dense_train_r2 = r2_score(y_train, y_pred_dense_train)
dense_train_ground_truth_mean = np.mean(y_train)
dense_train_rrmse = 100*dense_train_rmse/dense_train_ground_truth_mean

print('MSE: ', dense_train_mse)
print('RMSE: ', dense_train_rmse)
print('R2: ', dense_train_r2)
print('RRMSE: ', dense_train_rrmse)


# In[308]:


# Test MSE, RMSE, R2
dense_test_mse=mean_squared_error(y_test, y_pred_dense_test)
dense_test_rmse=np.sqrt(mean_squared_error(y_test, y_pred_dense_test))
dense_test_r2 = r2_score(y_test, y_pred_dense_test)
dense_test_ground_truth_mean = np.mean(y_test)
dense_test_rrmse = 100*dense_test_rmse/dense_test_ground_truth_mean

print('MSE: ', dense_test_mse)
print('RMSE: ', dense_test_rmse)
print('R2: ', dense_test_r2)
print('RRMSE: ', dense_test_rrmse)


# In[309]:


## Plot yield predictions
## What's the initial results you can get.
plt.figure(figsize=(14,6))

plt.subplot(1,2,1, label='Estimated Yield')
# Plot training & validation loss values
scatter=sns.regplot(x=y_train, y=y_pred_dense_train, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Training)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")
text_box = AnchoredText('MSE='+str(round(dense_train_mse,3))+'\n''RMSE='+str(round(dense_train_rmse,3))+'\n''RRMSE='+str(round(dense_train_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(dense_train_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)


plt.subplot(1,2,2)
# Plot training & validation accuracy values
scatter=sns.regplot(x=y_test, y=y_pred_dense_test, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Testing)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
text_box = AnchoredText('MSE='+str(round(dense_test_mse,3))+'\n''RMSE='+str(round(dense_test_rmse,3))+'\n''RRMSE='+str(round(dense_test_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(dense_test_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)

##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\dense_model_stack_pca.png', dpi=300,bbox_inches='tight')
plt.show()


# In[310]:


## Plot coefficients values
a = residuals_dense_train
b= residuals_dense_test

plt.figure(figsize=(14,6))
plt.subplot(1,2,1, label='Estimated Yield')
sns.distplot(a, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(a, bins=50,color= 'lightgreen',edgecolor='black', linewidth=1.2)
# the histogram of the data
plt.title('Error Distribution (Training)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.subplot(1,2,2)
sns.distplot(b, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(b, bins=50 ,color= 'lightgreen',edgecolor='black', linewidth=1.2)
plt.title('Error Distribution (Testing)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\dense_stack_residual_plots_pca.png', dpi=300,bbox_inches='tight')
plt.show()


# In[311]:


from keras.applications.resnet_v2 import ResNet152V2


# In[ ]:


input_shape = 32
batch_size = 128
epochs = 200
learning_rate = 0.0001


# In[312]:


def get_resnet152v2(input_shape, learning_rate):
    # create the base pre-trained model
    resnet_152_v2_model = ResNet152V2(include_top=False, weights= None, input_shape=(input_shape, input_shape, 1))
    x = resnet_152_v2_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='linear')(x)
    model = Model(inputs=resnet_152_v2_model.input, outputs=predictions)

    for layer in resnet_152_v2_model.layers:
        layer.trainable = True

    opt = optimizers.Adam(lr=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[keras.metrics.RootMeanSquaredError()])

    return model


# In[313]:


model.summary()


# In[314]:


## Define model callbacks
keras_callbacks   = [
      EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.0001),
                      LearningRateScheduler(lr_time_based_decay, verbose=1)
]


# In[315]:


history=model.fit(train_generator,
                    validation_data=(val_generator),
                    steps_per_epoch = train_generator.n//train_generator.batch_size,
                    validation_steps = val_generator.n//val_generator.batch_size,
                    epochs=epochs,callbacks=keras_callbacks)


# In[316]:


## Plot training loss
import matplotlib.pyplot as plt
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Training & Validation Loss', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True)
text_box = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)


plt.subplot(1,2,2)
# Plot training & validation accuracy values
plt.plot(history.history['root_mean_squared_error'], label='Train')
plt.plot(history.history['val_root_mean_squared_error'], label='Validation')
plt.title('Training & Validation RMSE', fontsize=15)
plt.ylabel('Epoch', fontsize=15)
plt.xlabel('RMSE', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True)
text_box = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.tight_layout()
#plt.savefig(r'D:\soybean_yield\figures\adrian_csci4760_project5_model_1_loss_graphs.png', dpi=300)
plt.show()


# In[317]:


## Get training and testing loss and accuracy
train_loss_1, train_acc_1 = model.evaluate(train_generator)
test_loss_1, test_acc_1 = model.evaluate(test_generator)

print('Training Accuracy: ', train_acc_1)
print('Training Loss: ', train_loss_1)
print('Testing Accuracy: ', test_acc_1)
print('Testing Loss: ', test_loss_1)


# In[318]:


## Predict on train and test data
y_pred_dense_train = model.predict(train_img_pca_2d)
y_pred_dense_test = model.predict(test_img_pca_2d)


# In[319]:


## get model error (residuals)
residuals_dense_train= y_train - np.squeeze(y_pred_dense_train)
residuals_dense_test= y_test - np.squeeze(y_pred_dense_test)


# In[323]:


# Train MSE, RMSE, R2
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

dense_train_mse=mean_squared_error(y_train, y_pred_dense_train)
dense_train_rmse=np.sqrt(mean_squared_error(y_train, y_pred_dense_train))
dense_train_r2 = r2_score(y_train, y_pred_dense_train)
dense_train_ground_truth_mean = np.mean(y_train)
dense_train_rrmse = 100*dense_train_rmse/dense_train_ground_truth_mean

print('MSE: ', dense_train_mse)
print('RMSE: ', dense_train_rmse)
print('R2: ', dense_train_r2)
print('RRMSE: ', dense_train_rrmse)


# In[324]:


# Test MSE, RMSE, R2
dense_test_mse=mean_squared_error(y_test, y_pred_dense_test)
dense_test_rmse=np.sqrt(mean_squared_error(y_test, y_pred_dense_test))
dense_test_r2 = r2_score(y_test, y_pred_dense_test)
dense_test_ground_truth_mean = np.mean(y_test)
dense_test_rrmse = 100*dense_test_rmse/dense_test_ground_truth_mean

print('MSE: ', dense_test_mse)
print('RMSE: ', dense_test_rmse)
print('R2: ', dense_test_r2)
print('RRMSE: ', dense_test_rrmse)


# In[325]:


## Plot yield predictions
## What's the initial results you can get.
plt.figure(figsize=(14,6))

plt.subplot(1,2,1, label='Estimated Yield')
# Plot training & validation loss values
scatter=sns.regplot(x=y_train, y=y_pred_dense_train, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Training)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")
text_box = AnchoredText('MSE='+str(round(dense_train_mse,3))+'\n''RMSE='+str(round(dense_train_rmse,3))+'\n''RRMSE='+str(round(dense_train_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(dense_train_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)


plt.subplot(1,2,2)
# Plot training & validation accuracy values
scatter=sns.regplot(x=y_test, y=y_pred_dense_test, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Testing)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
text_box = AnchoredText('MSE='+str(round(dense_test_mse,3))+'\n''RMSE='+str(round(dense_test_rmse,3))+'\n''RRMSE='+str(round(dense_test_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(dense_test_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)

##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\resnet_model_stack_pca.png', dpi=300,bbox_inches='tight')
plt.show()


# In[322]:


## Plot coefficients values
a = residuals_dense_train
b= residuals_dense_test

plt.figure(figsize=(14,6))
plt.subplot(1,2,1, label='Estimated Yield')
sns.distplot(a, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(a, bins=50,color= 'lightgreen',edgecolor='black', linewidth=1.2)
# the histogram of the data
plt.title('Error Distribution (Training)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.subplot(1,2,2)
sns.distplot(b, color='red',hist_kws={"color":"green","edgecolor": 'black'},rug=False)
#plt.hist(b, bins=50 ,color= 'lightgreen',edgecolor='black', linewidth=1.2)
plt.title('Error Distribution (Testing)', fontsize=20);
plt.ylabel('Count', fontsize=16)
plt.xlabel('Residual Value', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
text_box = AnchoredText('B.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)

plt.tight_layout()
plt.savefig(r'D:\soybean_yield\figures\resnet_stack_residual_plots_pca.png', dpi=300,bbox_inches='tight')
plt.show()


# In[115]:


## Compare networks
## Load metrics from part 1

col_name = ['MSE', 'RMSE','RRMSE','$R^2$ Score']
row_name= ['PLSR','DT','RF','SVR','NN','NN2']

metrics_df=pd.DataFrame([[pls_train_mse, pls_train_rmse, pls_train_rrmse,pls_train_r2],
                         [dt_train_mse, dt_train_rmse, dt_train_rrmse,dt_train_r2],
                         [rf_train_mse, rf_train_rmse, rf_train_rrmse,rf_train_r2],
                         [sv_train_mse, sv_train_rmse, sv_train_rrmse,sv_train_r2],
                         [nn_train_mse, nn_train_rmse, nn_train_rrmse,nn_train_r2],
                         [nn2_train_mse, nn2_train_rmse, nn2_train_rrmse,nn2_train_r2]], columns=col_name, index=row_name)

metrics_df=(metrics_df.style.highlight_min(axis=0, props='background-color:lightgreen;', subset=['MSE','RMSE','RRMSE'])
         .highlight_max(axis=0, props='background-color:lightgreen;', subset=['$R^2$ Score']))

metrics_df


# In[116]:


col_name = ['MSE', 'RMSE','RRMSE','$R^2$ Score']
row_name= ['PLSR','DT','RF','SVR','NN','NN2']

metrics_df=pd.DataFrame([[pls_test_mse, pls_test_rmse, pls_test_rrmse,pls_test_r2],
                         [dt_test_mse, dt_test_rmse, dt_test_rrmse,dt_test_r2],
                         [rf_test_mse, rf_test_rmse, rf_test_rrmse,rf_test_r2],
                         [sv_test_mse, sv_test_rmse, sv_test_rrmse,sv_test_r2],
                         [nn_test_mse, nn_test_rmse, nn_test_rrmse,nn_test_r2],
                         [nn2_test_mse, nn2_test_rmse, nn2_test_rrmse,nn2_test_r2]], columns=col_name, index=row_name)

metrics_df=(metrics_df.style.highlight_min(axis=0, props='background-color:lightgreen;', subset=['MSE','RMSE','RRMSE'])
         .highlight_max(axis=0, props='background-color:lightgreen;', subset=['$R^2$ Score']))

metrics_df


# In[117]:


# Load in the L3 soybean field shapefile
import geopandas as gpd
l3_plots = gpd.read_file('D:/soybean_yield/plot_boundary_shapefiles/L3_yield_plot_2017.shp')
l3_plots


# In[118]:


## Add an underscore to the plot ID for processing down the road
import pandas as pd
col_list = l3_plots['plot'].tolist()

new_list= []
for j in col_list:
    filename = str(j)
    filename = filename.replace("-", "_" )
    new_list.append(filename)
    column_names = ["plot_ID"]
    df = pd.DataFrame(new_list, columns = column_names)
    ##print(df)


# In[119]:


l3_plots['plot'] = df.plot_ID
l3_plots


# In[120]:


## Rename 'plot'
l3_plots= l3_plots.rename(columns={'plot': 'plot_ID'})
l3_plots


# In[121]:


## View the L3 shapefile
import matplotlib.pyplot as plt
l3_plots.plot(legend=True,figsize=(80,20))
plt.title('Bradford Soybean L3', fontsize=40);
plt.ylabel('Y', fontsize=36)
plt.xlabel('X', fontsize=36)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.show()


# In[136]:


res_df_train=pd.DataFrame([X_train, y_train, residuals_train_sv])
res_df_train = res_df_train.T
res_df_train.columns.values[2] = "residuals"
res_df_train['predicted_yield_kgha'] =y_pred_sv
res_df_train


# In[138]:


res_df_test=pd.DataFrame([X_test, y_test, residuals_test_sv])
res_df_test = res_df_test.T
res_df_test.columns.values[2] = "residuals"
res_df_test['predicted_yield_kgha'] =y_pred_sv_test
res_df_test


# In[139]:


frames = [res_df_train, res_df_test]
result = pd.concat(frames)
result


# In[140]:


test=result.sort_values(by=['image_path'])
test


# In[141]:


test.to_csv(r'D:\soybean_yield\csv\test.csv')


# In[142]:


## Import the soybean yield csv
import pandas as pd
test_df = pd.read_csv(r'D:\soybean_yield\csv\l3_residuals.csv')


# In[143]:


test_df


# In[144]:


test2=l3_plots.sort_values(by=['plot_ID'])
test2


# In[145]:


data_merged2 = pd.merge(test2, test_df)


# In[146]:


data_merged2


# In[147]:


## View plots by yield
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ax=data_merged2.plot(column='yield_kgha', cmap='Greens', legend=True,figsize=(80,20), legend_kwds={'label': "Yield Ha/Ac"})
plt.title('Bradford Soybean Yield L3', fontsize=40);
plt.ylabel('Y', fontsize=36)
plt.xlabel('X', fontsize=36)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
fig = ax.figure
cb_ax = fig.axes[1]
cb_ax.tick_params(labelsize=40)
##cb_ax.set_yticklabels({'fontsize': 30})
plt.show()


# In[148]:


## View plots by yield
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ax=data_merged2.plot(column='predicted_yield_kgha',cmap='Greens', legend=True,figsize=(80,20), legend_kwds={'label': "Yield Ha/Ac"})
plt.title('Bradford Soybean Predicted Yield L3', fontsize=40);
plt.ylabel('Y', fontsize=36)
plt.xlabel('X', fontsize=36)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
fig = ax.figure
cb_ax = fig.axes[1]
cb_ax.tick_params(labelsize=40)
##cb_ax.set_yticklabels({'fontsize': 30})
plt.show()


# In[153]:


## View plots by yield
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ax=data_merged2.plot(column='residuals',cmap='coolwarm', legend=True,figsize=(80,20), legend_kwds={'label': "Yield Ha/Ac"})
plt.title('Bradford Soybean Yield L3 Residual Values', fontsize=40);
plt.ylabel('Y', fontsize=36)
plt.xlabel('X', fontsize=36)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
fig = ax.figure
cb_ax = fig.axes[1]
cb_ax.tick_params(labelsize=40)
##cb_ax.set_yticklabels({'fontsize': 30})
plt.show()


# In[174]:


# Plot shapefile over raster if you want
import rasterio
import numpy as np
from rasterio import plot as rasterplot
import geopandas as gpd
from matplotlib import pyplot as plt


tiff = rasterio.open('D:/soybean_yield/macro_corpus/l3_macro_2017/clip/L3_MultiSpec_Parrot_20170909_flight3_clip.tif')
tiff_src=tiff.read(),
tiff_false= tiff_src[0:3],
tiff_extent = [tiff.bounds[0], tiff.bounds[2], tiff.bounds[1], tiff.bounds[3]]

#H1G_2020_src= H1G_2020_src.read()
#H1G_2020_false = H1G_2020_src[0:3] # read in red, green, blue

shapefile= data_merged2
f, ax = plt.subplots(figsize=(100,30))

# plot DEM
rasterplot.show(
    tiff.read((4,3,2)),  # use tiff.read(1) with your data
    #tiff_false,
    extent=tiff_extent,
    ax=ax,
    cmap='gray'

)

#shapefile.plot(ax=ax,column='yield_kgha', legend=True,figsize=(100,30), legend_kwds={'label': "Yield Ha/Ac"})
shapefile.plot(ax=ax,column='residuals',cmap='coolwarm',alpha=0.75, legend=True,figsize=(100,30), legend_kwds={'label': "Yield Ha/Ac"})
plt.title('Bradford Soybean Yield L3', fontsize=46);
plt.ylabel('Y', fontsize=38)
plt.xlabel('X', fontsize=38)
plt.xticks(fontsize=34)
plt.yticks(fontsize=34)
cb_ax = fig.axes[0]
cb_ax.tick_params(labelsize=34)
cb_ax.set_yticklabels({'fontsize': 34})
plt.show()


# In[ ]:
