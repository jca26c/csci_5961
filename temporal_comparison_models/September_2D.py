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
resize_list=[]
for i in file_list_train:
    image=gdal.Open(i).ReadAsArray()
    july_image= image [12:19,:,:]
    #image_rescale=img_trans/255
    image_t=july_image.transpose(2,1,0)
    image_resized = resize(image_t, (220,600,8),anti_aliasing=False)
    resize_list.append(image_resized)

train_data=np.stack(resize_list, axis=0)
print(train_data.shape)


# In[17]:


## Make sure the imagery looks right
plt.figure(1,(24,10)).subplots_adjust(hspace=0.5)
plt.subplot(411)
plt.title('Image 10')
plt.imshow(train_data[10,:,:,0],cmap='gray')

plt.subplot(412)
plt.title('Image 20')
plt.imshow(train_data[20,:,:,1],cmap='gray')

plt.subplot(413)
plt.title('Image 30')
plt.imshow(train_data[30,:,:,2],cmap='gray')

plt.subplot(414)
plt.title('Image 40')
plt.imshow(train_data[90,:,:,4],cmap='gray')

plt.savefig(r'D:\soybean_yield\figures\train_image_stack.png', dpi=300,bbox_inches='tight')
plt.show()


# In[18]:


# Create file list for training data
resize_list=[]
for i in file_list_val:
    image=gdal.Open(i).ReadAsArray()
    july_image= image [12:19,:,:]
    image_t=july_image.transpose(2,1,0)
    image_resized = resize(image_t, (220,600,8),anti_aliasing=False)
    resize_list.append(image_resized)

val_data=np.stack(resize_list, axis=0)
print(val_data.shape)


# In[19]:


## Make sure the imagery looks right
plt.figure(1,(24,10)).subplots_adjust(hspace=0.5)
plt.subplot(411)
plt.title('Image 10')
plt.imshow(val_data[10,:,:,0],cmap='gray')

plt.subplot(412)
plt.title('Image 20')
plt.imshow(val_data[20,:,:,1],cmap='gray')

plt.subplot(413)
plt.title('Image 30')
plt.imshow(val_data[30,:,:,3],cmap='gray')

plt.subplot(414)
plt.title('Image 40')
plt.imshow(val_data[40,:,:,4],cmap='gray')

plt.savefig(r'D:\soybean_yield\figures\val_image_stack.png', dpi=300,bbox_inches='tight')
plt.show()


# In[20]:


# Create file list for test data
resize_list=[]
for i in file_list_test:
    image=gdal.Open(i).ReadAsArray()
    july_image= image [12:19,:,:]
    #image_rescale=img_trans/255
    image_t=july_image.transpose(2,1,0)
    image_resized = resize(image_t, (220,600,8),anti_aliasing=False)
    resize_list.append(image_resized)

test_data=np.stack(resize_list, axis=0)
print(test_data.shape)


# In[21]:


## Make sure the imagery looks right
plt.figure(1,(24,10)).subplots_adjust(hspace=0.5)
plt.subplot(411)
plt.title('Image 10')
plt.imshow(test_data[10,:,:,0],cmap='gray')

plt.subplot(412)
plt.title('Image 20')
plt.imshow(test_data[50,:,:,1],cmap='gray')

plt.subplot(413)
plt.title('Image 30')
plt.imshow(test_data[60,:,:,2],cmap='gray')

plt.subplot(414)
plt.title('Image 40')
plt.imshow(test_data[70,:,:,3],cmap='gray')

plt.savefig(r'D:\soybean_yield\figures\stack_test_image.png', dpi=300,bbox_inches='tight')
plt.show()


# In[22]:


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


# In[23]:


# Print min/max values
print("training: ",np.min(train_data),np.max(train_data))
print("validation: ",np.min(val_data),np.max(val_data))
print("testing: ",np.min(test_data),np.max(test_data))


# In[24]:


#normalize all values to be between 0 and 1
train_data = (train_data-np.min(train_data))/(np.max(train_data)-np.min(train_data))
val_data = (val_data-np.min(val_data))/(np.max(val_data)-np.min(val_data))
test_data = (test_data-np.min(test_data))/(np.max(test_data)-np.min(test_data))                                           


# In[25]:


# Print min/max values
print("training: ",np.min(train_data),np.max(train_data))
print("validation ",np.min(val_data),np.max(val_data))
print("testing: ",np.min(test_data),np.max(test_data))


# In[26]:


# Print data shape
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)


# In[27]:


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


# In[28]:


print("Tensorflow-version:", tensorflow.__version__)


# In[29]:


first_model = DenseNet121(weights = None, input_shape=(220, 600, 8), include_top=False)


# In[30]:


x=first_model.output
x= GlobalAveragePooling2D()(x)
x= BatchNormalization()(x)
x= Dropout(0.5)(x)
x= Dense(1024,activation='relu')(x) 
x= Dense(512,activation='relu')(x) 
x= BatchNormalization()(x)
x= Dropout(0.5)(x)
x= Flatten()(x)
preds=Dense(1,activation='linear')(x) #FC-layer


# In[31]:


model=Model(inputs=first_model.input,outputs=preds)
model.summary()


# In[32]:


for layer in model.layers[:-9]:
    layer.trainable=False
    
for layer in model.layers[-9:]:
    layer.trainable=True


# In[33]:


# compile model 
opt = Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=[keras.metrics.RootMeanSquaredError()])


# In[34]:


## Use learning rate decay to improve model performance
epochs=200
batch_size= 128
initial_learning_rate = 0.001
decay = initial_learning_rate / epochs

def lr_time_based_decay(epoch, lr):
    return lr * 1 / (1 + decay * epoch)


# In[35]:


## Define model callbacks
keras_callbacks   = [
      EarlyStopping(monitor='val_loss', patience=20, mode='min', min_delta=0.0001),
    LearningRateScheduler(lr_time_based_decay, verbose=1)
]


# In[36]:


datagen = ImageDataGenerator()


# In[37]:


train_generator= datagen.flow(train_data, y_train)
val_generator= datagen.flow(val_data, y_val)
test_generator= datagen.flow(test_data, y_test)


# In[38]:


history=model.fit(train_generator,
                    validation_data=(val_generator),
                    steps_per_epoch = train_generator.n//train_generator.batch_size,
                    validation_steps = val_generator.n//val_generator.batch_size,
                    epochs=epochs,callbacks=keras_callbacks)


# In[39]:


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


# In[40]:


## Get training and testing loss and accuracy
dense_train_loss_1, dense_train_acc_1 = model.evaluate(train_generator)
dense_test_loss_1, dense_test_acc_1 = model.evaluate(test_generator)

print('Training Accuracy: ', dense_train_acc_1)
print('Training Loss: ', dense_train_loss_1)
print('Testing Accuracy: ', dense_test_acc_1)
print('Testing Loss: ', dense_test_loss_1)


# In[41]:


## Predict on train and test data
y_pred_dense_train = model.predict(train_data)
y_pred_dense_test = model.predict(test_data)


# In[42]:


## get model error (residuals)
residuals_dense_train= y_train - np.squeeze(y_pred_dense_train)
residuals_dense_test= y_test - np.squeeze(y_pred_dense_test)


# In[43]:


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


# In[44]:


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


# In[45]:


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


# In[46]:


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


# In[47]:


## Train ResNet
from keras.applications.resnet_v2 import ResNet152V2


# In[48]:


input_shape = 32
batch_size = 128
epochs = 200
learning_rate = 0.0001


# In[49]:


def get_resnet152v2(input_shape, learning_rate):
    # create the base pre-trained model
    resnet_152_v2_model = ResNet152V2(include_top=False, weights= None, input_shape=(220, 600, 8))
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


# In[50]:


from keras.applications.resnet_v2 import preprocess_input
model2= get_resnet152v2(input_shape, learning_rate)


# In[51]:


model2.summary()


# In[ ]:


history2=model2.fit(train_generator,
                    validation_data=(val_generator),
                    steps_per_epoch = train_generator.n//train_generator.batch_size,
                    validation_steps = val_generator.n//val_generator.batch_size,
                    epochs=epochs,callbacks=keras_callbacks)


# In[ ]:


## Plot training loss
import matplotlib.pyplot as plt
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
# Plot training & validation loss values
plt.plot(history2.history['loss'], label='Train')
plt.plot(history2.history['val_loss'], label='Validation')
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
plt.plot(history2.history['root_mean_squared_error'], label='Train')
plt.plot(history2.history['val_root_mean_squared_error'], label='Validation')
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


# In[ ]:


## Get training and testing loss and accuracy
res_train_loss_1, res_train_acc_1 = model2.evaluate(train_generator)
res_test_loss_1, res_test_acc_1 = model2.evaluate(test_generator)

print('Training Accuracy: ', res_train_acc_1)
print('Training Loss: ', res_train_loss_1)
print('Testing Accuracy: ', res_test_acc_1)
print('Testing Loss: ', res_test_loss_1)


# In[ ]:


## Predict on train and test data
y_pred_res_train = model2.predict(train_data)
y_pred_res_test = model2.predict(test_data)


# In[ ]:


## get model error (residuals)
residuals_res_train= y_train - np.squeeze(y_pred_res_train)
residuals_res_test= y_test - np.squeeze(y_pred_res_test)


# In[ ]:


# Train MSE, RMSE, R2
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

res_train_mse=mean_squared_error(y_train, y_pred_res_train)
res_train_rmse=np.sqrt(mean_squared_error(y_train, y_pred_res_train))
res_train_r2 = r2_score(y_train, y_pred_res_train)
res_train_ground_truth_mean = np.mean(y_train)
res_train_rrmse = 100*res_train_rmse/res_train_ground_truth_mean

print('MSE: ', res_train_mse)
print('RMSE: ', res_train_rmse)
print('R2: ', res_train_r2)
print('RRMSE: ', res_train_rrmse)


# In[ ]:


# Test MSE, RMSE, R2
res_test_mse=mean_squared_error(y_test, y_pred_res_test)
res_test_rmse=np.sqrt(mean_squared_error(y_test, y_pred_res_test))
res_test_r2 = r2_score(y_test, y_pred_res_test)
res_test_ground_truth_mean = np.mean(y_test)
res_test_rrmse = 100*res_test_rmse/res_test_ground_truth_mean

print('MSE: ', res_test_mse)
print('RMSE: ', res_test_rmse)
print('R2: ', res_test_r2)
print('RRMSE: ', res_test_rrmse)


# In[ ]:


## Plot yield predictions
## What's the initial results you can get.
plt.figure(figsize=(14,6))

plt.subplot(1,2,1, label='Estimated Yield')
# Plot training & validation loss values
scatter=sns.regplot(x=y_train, y=y_pred_res_train, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Training)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
##plt.legend(loc='upper right', fontsize=12,title="Estimated Yield")
text_box = AnchoredText('MSE='+str(round(res_train_mse,3))+'\n''RMSE='+str(round(res_train_rmse,3))+'\n''RRMSE='+str(round(res_train_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(res_train_r2,3)), frameon=True, loc=4, pad=0.5)
plt.setp(text_box.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box)
plt.grid(True)
text_box2 = AnchoredText('A.)', frameon=False, loc=1, pad=0.5, prop={'color':'k','fontsize':16,
                                   'fontfamily':'DejaVu Sans'})
plt.setp(text_box2.patch, facecolor='white', alpha=0.5)
plt.gca().add_artist(text_box2)


plt.subplot(1,2,2)
# Plot training & validation accuracy values
scatter=sns.regplot(x=y_test, y=y_pred_res_test, color='g',line_kws={"color": "red"})
plt.title('Measured vs Estimated Yield (Testing)', fontsize=15)
plt.ylabel('Estimated Yield (Kg/Ha)', fontsize=15)
plt.xlabel('Measured Yield (Kg/Ha)', fontsize=15)
plt.xticks( fontsize=12)
plt.yticks( fontsize=12)
text_box = AnchoredText('MSE='+str(round(res_test_mse,3))+'\n''RMSE='+str(round(res_test_rmse,3))+'\n''RRMSE='+str(round(res_test_rrmse,3))+
                        '%'+'\n' 'r$^{2}$='+str(round(res_test_r2,3)), frameon=True, loc=4, pad=0.5)
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


# In[ ]:


## Plot coefficients values
a = residuals_res_train
b= residuals_res_test

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


# In[ ]:
