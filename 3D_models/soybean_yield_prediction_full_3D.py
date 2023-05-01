#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.getcwd())


# In[2]:


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


# In[3]:


## Import the soybean yield csv
import pandas as pd
yield_df = pd.read_csv(r'D:\soybean_yield\csv\yield_result_stacks_gold.csv')


# In[4]:


## Display dataframe to get a sense of the whole dataset
yield_df


# In[5]:


## Round dataframe to two decimals places
yield_df=yield_df.round(2)
yield_df


# In[6]:


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


# In[7]:


## Shuffle the data to get a random sample
data_shuffled=yield_df.sample(frac=1).reset_index(drop=True)
data_shuffled


# In[8]:


## Check if NANs exist
data_shuffled.isnull().values.any()


# In[9]:


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


# In[10]:


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


# In[11]:


## Split the dataset into training and testing
from sklearn.model_selection import train_test_split
X=data_shuffled['image_path'].values
y=data_shuffled['yield_kgha'].values

## Split training and testing
X_train, X_test, y_train, y_test = train_test_split(data_shuffled['image_path'], data_shuffled['yield_kgha'], test_size=0.20, random_state=0,shuffle=False)


# In[12]:


## Split training and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


# In[13]:


## Print out the sizes of training and testing data
print("training data size: ",X_train.size)
print("validation data size: ",X_val.size)
print("test data size: ",X_test.size)


# In[14]:


## Print out the sizes of training and testing data
print("training data size: ",y_train.size)
print("validation data size: ",y_val.size)
print("test data size: ",y_test.size)


# In[15]:


## Create a list of .tif file
import os 
import glob
file_list_train = X_train
file_list_val = X_val
file_list_test = X_test


# In[16]:


# Create file list for training data
july_resize_list=[]
aug1_resize_list=[]
aug2_resize_list=[]
sept1_resize_list=[]
sept2_resize_list=[]

for i in file_list_train:
    image=gdal.Open(i).ReadAsArray()
    image_t=image.transpose(2,1,0)
    july_image= image_t [:,:,0:3]
    aug_image1= image_t [:,:,4:7]
    aug_image2= image_t [:,:,8:11]
    sept_image1= image_t [:,:,12:15]
    sept_image2= image_t [:,:,16:19]
    #image_rescale=img_trans/255
    image_resized_july = resize(july_image, (220,600,4),anti_aliasing=False)
    image_resized_aug1 = resize(aug_image1, (220,600,4),anti_aliasing=False)
    image_resized_aug2 = resize(aug_image2, (220,600,4),anti_aliasing=False)
    image_resized_sept1 = resize(sept_image1, (220,600,4),anti_aliasing=False)
    image_resized_sept2 = resize(sept_image2, (220,600,4),anti_aliasing=False)
    july_resize_list.append(image_resized_july)
    aug1_resize_list.append(image_resized_aug1)
    aug2_resize_list.append(image_resized_aug2)
    sept1_resize_list.append(image_resized_sept1)
    sept2_resize_list.append(image_resized_sept2)
    #resize_list= july_resize_list+ aug1_resize_list + aug2_resize_list + sept1_resize_list + sept2_resize_list

july_stack=np.stack(july_resize_list, axis=0)
aug1_stack=np.stack(aug1_resize_list, axis=0)
aug2_stack=np.stack(aug2_resize_list, axis=0)
sep1_stack=np.stack(sept1_resize_list, axis=0)
sep2_stack=np.stack(sept2_resize_list, axis=0)
train_data=np.stack((july_stack, aug1_stack, aug2_stack, sep1_stack, sep2_stack), axis=0)
#val_data=np.stack(resize_list, axis=0)
print(train_data.shape)


# In[17]:


test=train_data.transpose(1,0,2,3,4)
print(test.shape)


# In[18]:


## Make sure the imagery looks right
plt.figure(1,(24,10)).subplots_adjust(hspace=0.5)
plt.subplot(411)
plt.title('Image 10')
plt.imshow(train_data[1,0,:,:,0],cmap='gray')

plt.subplot(412)
plt.title('Image 20')
plt.imshow(train_data[1,1,:,:,1],cmap='gray')

plt.subplot(413)
plt.title('Image 30')
plt.imshow(train_data[1,2,:,:,2],cmap='gray')

plt.subplot(414)
plt.title('Image 40')
plt.imshow(train_data[1,3,:,:,3],cmap='gray')

plt.savefig(r'D:\soybean_yield\figures\3D_train_image_stack.png', dpi=300,bbox_inches='tight')
plt.show()


# In[19]:


# Create file list for training data
july_resize_list=[]
aug1_resize_list=[]
aug2_resize_list=[]
sept1_resize_list=[]
sept2_resize_list=[]

for i in file_list_val:
    image=gdal.Open(i).ReadAsArray()
    image_t=image.transpose(2,1,0)
    july_image= image_t [:,:,0:3]
    aug_image1= image_t [:,:,4:7]
    aug_image2= image_t [:,:,8:11]
    sept_image1= image_t [:,:,12:15]
    sept_image2= image_t [:,:,16:19]
    #image_rescale=img_trans/255
    image_resized_july = resize(july_image, (220,600,4),anti_aliasing=False)
    image_resized_aug1 = resize(aug_image1, (220,600,4),anti_aliasing=False)
    image_resized_aug2 = resize(aug_image2, (220,600,4),anti_aliasing=False)
    image_resized_sept1 = resize(sept_image1, (220,600,4),anti_aliasing=False)
    image_resized_sept2 = resize(sept_image2, (220,600,4),anti_aliasing=False)
    july_resize_list.append(image_resized_july)
    aug1_resize_list.append(image_resized_aug1)
    aug2_resize_list.append(image_resized_aug2)
    sept1_resize_list.append(image_resized_sept1)
    sept2_resize_list.append(image_resized_sept2)
    #resize_list= july_resize_list+ aug1_resize_list + aug2_resize_list + sept1_resize_list + sept2_resize_list

july_stack=np.stack(july_resize_list, axis=0)
aug1_stack=np.stack(aug1_resize_list, axis=0)
aug2_stack=np.stack(aug2_resize_list, axis=0)
sep1_stack=np.stack(sept1_resize_list, axis=0)
sep2_stack=np.stack(sept2_resize_list, axis=0)
val_data=np.stack((july_stack, aug1_stack, aug2_stack, sep1_stack, sep2_stack), axis=0)
#val_data=np.stack(resize_list, axis=0)
print(val_data.shape)


# In[20]:


## Make sure the imagery looks right
plt.figure(1,(24,10)).subplots_adjust(hspace=0.5)
plt.subplot(411)
plt.title('Image 10')
plt.imshow(val_data[1,0,:,:,0],cmap='gray')

plt.subplot(412)
plt.title('Image 20')
plt.imshow(val_data[1,0,:,:,1],cmap='gray')

plt.subplot(413)
plt.title('Image 30')
plt.imshow(val_data[1,0,:,:,2],cmap='gray')

plt.subplot(414)
plt.title('Image 40')
plt.imshow(val_data[1,0,:,:,3],cmap='gray')

plt.savefig(r'D:\soybean_yield\figures\3D_val_image_stack.png', dpi=300,bbox_inches='tight')
plt.show()


# In[21]:


# Create file list for training data
july_resize_list=[]
aug1_resize_list=[]
aug2_resize_list=[]
sept1_resize_list=[]
sept2_resize_list=[]

for i in file_list_test:
    image=gdal.Open(i).ReadAsArray()
    image_t=image.transpose(2,1,0)
    july_image= image_t [:,:,0:3]
    aug_image1= image_t [:,:,4:7]
    aug_image2= image_t [:,:,8:11]
    sept_image1= image_t [:,:,12:15]
    sept_image2= image_t [:,:,16:19]
    #image_rescale=img_trans/255
    image_resized_july = resize(july_image, (220,600,4),anti_aliasing=False)
    image_resized_aug1 = resize(aug_image1, (220,600,4),anti_aliasing=False)
    image_resized_aug2 = resize(aug_image2, (220,600,4),anti_aliasing=False)
    image_resized_sept1 = resize(sept_image1, (220,600,4),anti_aliasing=False)
    image_resized_sept2 = resize(sept_image2, (220,600,4),anti_aliasing=False)
    july_resize_list.append(image_resized_july)
    aug1_resize_list.append(image_resized_aug1)
    aug2_resize_list.append(image_resized_aug2)
    sept1_resize_list.append(image_resized_sept1)
    sept2_resize_list.append(image_resized_sept2)
    #resize_list= july_resize_list+ aug1_resize_list + aug2_resize_list + sept1_resize_list + sept2_resize_list

july_stack=np.stack(july_resize_list, axis=0)
aug1_stack=np.stack(aug1_resize_list, axis=0)
aug2_stack=np.stack(aug2_resize_list, axis=0)
sep1_stack=np.stack(sept1_resize_list, axis=0)
sep2_stack=np.stack(sept2_resize_list, axis=0)
test_data=np.stack((july_stack, aug1_stack, aug2_stack, sep1_stack, sep2_stack), axis=0)
#val_data=np.stack(resize_list, axis=0)
print(test_data.shape)


# In[22]:


## Make sure the imagery looks right
plt.figure(1,(24,10)).subplots_adjust(hspace=0.5)
plt.subplot(411)
plt.title('Image 10')
plt.imshow(test_data[1,0,:,:,0],cmap='gray')

plt.subplot(412)
plt.title('Image 20')
plt.imshow(test_data[1,0,:,:,1],cmap='gray')

plt.subplot(413)
plt.title('Image 30')
plt.imshow(test_data[1,0,:,:,2],cmap='gray')

plt.subplot(414)
plt.title('Image 40')
plt.imshow(test_data[1,0,:,:,3],cmap='gray')

plt.savefig(r'D:\soybean_yield\figures\3D_test_image_stack.png', dpi=300,bbox_inches='tight')
plt.show()


# In[23]:


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


# In[24]:


# Print min/max values
print("training: ",np.min(train_data),np.max(train_data))
print("validation: ",np.min(val_data),np.max(val_data))
print("testing: ",np.min(test_data),np.max(test_data))


# In[25]:


#normalize all values to be between 0 and 1
train_data = (train_data-np.min(train_data))/(np.max(train_data)-np.min(train_data))
val_data = (val_data-np.min(val_data))/(np.max(val_data)-np.min(val_data))
test_data = (test_data-np.min(test_data))/(np.max(test_data)-np.min(test_data))                                           


# In[26]:


# Print min/max values
print("training: ",np.min(train_data),np.max(train_data))
print("validation ",np.min(val_data),np.max(val_data))
print("testing: ",np.min(test_data),np.max(test_data))


# In[27]:


# Print data shape
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)


# In[28]:


train_data=train_data.transpose(1,0,2,3,4)
val_data=val_data.transpose(1,0,2,3,4)
test_data=test_data.transpose(1,0,2,3,4)


# In[29]:


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
from tensorflow.keras.layers import ReLU, concatenate
from keras import optimizers
from keras.layers import Conv3D
from tensorflow.keras.layers import Flatten,MaxPooling3D,Dropout,AvgPool3D,GlobalAveragePooling3D,ZeroPadding3D
from tensorflow.keras.layers import ReLU, concatenate
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# In[30]:


print("Tensorflow-version:", tensorflow.__version__)


# In[33]:


model = Sequential()
model.add(layers.Conv3D(32,(3,3,3),activation='relu',input_shape=(5,220, 600, 4),bias_initializer=Constant(0.01)))
model.add(layers.Conv3D(64,(3,3,3),activation='relu',bias_initializer=Constant(0.01)))
model.add(layers.Conv3D(64,(1,1,1),activation='relu',bias_initializer=Constant(0.01)))
#model.add(layers.MaxPooling3D((2,2,2),padding='same'))
#model.add(layers.Conv3D(64,(3,3,3),activation='relu'))
#model.add(layers.Conv3D(64,(2,2,2),activation='relu'))
model.add(layers.MaxPooling3D((2,2,2),padding='same'))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(512,'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256,'relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,'linear'))
model.summary()


# In[34]:


# compile model 
opt = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=[keras.metrics.RootMeanSquaredError()])


# In[35]:


## Use learning rate decay to improve model performance
epochs=200
batch_size= 64
initial_learning_rate = 0.001
decay = initial_learning_rate / epochs

def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)


# In[36]:


## Define model callbacks
keras_callbacks   = [
      EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.0001),
      ModelCheckpoint('D:/soybean_yield/callbacks/checkmodel.model_1', monitor='val_loss', save_best_only=True, mode='min'),
                      LearningRateScheduler(lr_time_based_decay, verbose=1)
]


# In[37]:


history = model.fit(train_data, y_train, epochs=epochs, batch_size=batch_size, validation_data=(val_data, y_val), verbose=1, callbacks=keras_callbacks) 


# In[38]:


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


# In[39]:


## Get training and testing loss and accuracy
dense_train_loss_1, dense_train_acc_1 = model.evaluate(train_data,y_train)
dense_test_loss_1, dense_test_acc_1 = model.evaluate(test_data,y_test)

print('Training Accuracy: ', dense_train_acc_1)
print('Training Loss: ', dense_train_loss_1)
print('Testing Accuracy: ', dense_test_acc_1)
print('Testing Loss: ', dense_test_loss_1)


# In[40]:


## Predict on train and test data
y_pred_dense_train = model.predict(train_data)
y_pred_dense_test = model.predict(test_data)


# In[41]:


## get model error (residuals)
residuals_dense_train= y_train - np.squeeze(y_pred_dense_train)
residuals_dense_test= y_test - np.squeeze(y_pred_dense_test)


# In[42]:


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


# In[43]:


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


# In[44]:


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
plt.savefig(r'D:\soybean_yield\figures\dense_model_stack_3D.png', dpi=300,bbox_inches='tight')
plt.show()


# In[45]:


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
plt.savefig(r'D:\soybean_yield\figures\dense_stack_residual_plots_3D.png', dpi=300,bbox_inches='tight')
plt.show()


# In[62]:


# serialize model to JSON
model_json = model.to_json()
with open(r"D:\soybean_yield\model_results\three_dee_cnn_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(r'D:\soybean_yield\model_results\three_dee_cnn_model.h5')
print("Saved model to disk")


# In[63]:


# load json and create model
from tensorflow.keras.models import Sequential, model_from_json
json_file = open(r'D:\soybean_yield\model_results\three_dee_cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"D:\soybean_yield\model_results\three_dee_cnn_model.h5")
print("Loaded model from disk")


# In[66]:


loaded_model.compile(loss='mean_squared_error', optimizer=opt, metrics=[keras.metrics.RootMeanSquaredError()])
#result = loaded_model.score(test_data, y_test)


# In[67]:


score = loaded_model.evaluate(test_data, y_test, verbose=0)


# In[68]:


score
