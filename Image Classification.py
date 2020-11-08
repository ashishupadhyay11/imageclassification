
# Data : Images
# Use Python libraries to scrape the data

!pip install bing-image-downloader

#create new directory
!mkdir Images

#import downloader
from bing_image_downloader import downloader 
downloader.download("pretty sunflower", limit=30,output_dir='Images',adult_filter_off=True)

downloader.download("dragon", limit=30,output_dir='Images',adult_filter_off=True)

downloader.download("ice cream cone", limit=30,output_dir='Images',adult_filter_off=True)

downloader.download("rugby ball leather", limit=30,output_dir='Images',adult_filter_off=True)

#load library which gives execution time for each cell
!pip install ipython-autotime
%load_ext autotime

# Preprocessing
# 1. Resize
# 2. Flatten

import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize

#use -flatten()to convert matrix to vector

target = []
images = []
flat_data = []

DATADIR = '/content/Images'
CATEGORIES = ['pretty sunflower', 'dragon', 'rugby ball leather', 'ice cream cone']

for category in CATEGORIES:
  class_num = CATEGORIES.index(category) #Label encoding the values
  path = os.path.join(DATADIR, category) #Create path for all images

  for img in os.listdir(path):
    img_array = imread(os.path.join(path, img))
    #print(img_array.shape)
    #plt.imshow(img_array)
    img_resized = resize(img_array,(150,150,3)) #Normalizes the value from 0 to 1
    flat_data.append(img_resized.flatten())
    images.append(img_resized)
    target.append(class_num)

flat_data = np.array(flat_data)
target = np.array(target)
images = np.array(images)

unique,count = np.unique(target, return_counts=True)
plt.bar(CATEGORIES, count)

# Split data into training and testing 

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(flat_data,target,test_size=0.3,random_state=109) #Split arrays or matrices into random train and test subsets

from sklearn.model_selection import GridSearchCV
from sklearn import svm
# Documentation https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
param_grid = [
              {'C':[1,10,100,1000],'kernel':['linear']},
              {'C':[1,10,100,1000],'gamma':[0.001, 0.0001],'kernel':['rbf']}
]
svc = svm.SVC(probability=True) #Parameter to show all possible outcomes (% match)
clf = GridSearchCV(svc,param_grid)
clf.fit(x_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix
y_pred = clf.predict(x_test)
accuracy_score(y_pred,y_test) # % acuracy in result

confusion_matrix(y_pred,y_test) # Shows varince in predicted and actual output

# Save the model using pickle library for deployment

import pickle
pickle.dump(clf, open('img_model.p','wb'))

model = pickle.load(open('img_model.p','rb'))

# Testing a brand new image

flat_data = []
url = input('Enter your URL')
img = imread(url)
img_resized = resize(img,(150,150,3))
flat_data.append(img_resized.flatten())
flat_data = np.array(flat_data)
print(img.shape)
plt.imshow(img_resized)
y_out = model.predict(flat_data)
y_out = CATEGORIES[y_out[0]]
print(f' PREDICTED OUTPUT: {y_out}')

# Deployment (WebApp)

!pip install streamlit

!pip install pyngrok
from pyngrok import ngrok

%%writefile app.py
import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import pickle
from PIL import Image
st.title('Image Classifier using Machine Learning')
st.text('Upload the Image')

model = pickle.load(open('img_model.p','rb'))

uploaded_file = st.file_uploader("Choose an image...", type ="jpg")
if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img,caption='Uploaded Image')

  if st.button('PREDICT'):
    CATEGORIES = ['pretty sunflower', 'dragon', 'rugby ball leather', 'ice cream cone']
    st.write('Result...')
    flat_data = []
    img = np.array(img)
    img_resized = resize(img,(150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    print(img.shape)
    y_out = model.predict(flat_data)
    y_out = CATEGORIES[y_out[0]]
    st.title(f' PREDICTED OUTPUT: {y_out}')
    q = model.predict_proba(flat_data)
    for index, item in enumerate(CATEGORIES):
      st.write(f'{item}:{q[0][index]*100}%')

!nohup streamlit run app.py &

url = ngrok.connect('8051')
url