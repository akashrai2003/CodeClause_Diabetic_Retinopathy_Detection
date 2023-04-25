import numpy as np
import tensorflow as tf
import keras
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras import backend as K
from keras.layers.core import Dense,Dropout
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import itertools
import tkinter as tk
from tkinter import filedialog,messagebox
import cv2
import matplotlib.pyplot as plt
plt.rcdefaults()
from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.utils import img_to_array
from itertools import chain

Disease_classes={
    0:'No_DR',
    1:'Mild',
    2:'Moderate',
    3:'Severe',
    4:'Proliferate_DR'
}

def load_training_model():
    mobile=keras.applications.mobilenet.MobileNet()
    x=mobile.layers[-6].output
    x=Dropout(0.25)(x)
    x=GlobalAveragePooling2D()(x)
    predictions=Dense(5,activation='softmax')(x)
    model=Model(inputs=mobile.input,outputs=predictions)
    
    for layer in model.layers[:-23]:
        layer.trainable=False
    
    model.load_weights('model.h5')
    return model

def predict_test_image_file(model):
    root=tk.Tk()
    root.withdraw()
    imageFileNames=filedialog.askopenfilenames()

    image=cv2.imread(imageFileNames)

    image=cv2.resize(image,(224,224))
    image=image.astype('float')/255.0
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)

    predictions=model.predict(image)
    print(predictions)

    final_prediction=predictions.argmax(axis=1)[0]
    print(final_prediction)

    disease_name=Disease_classes[final_prediction]
    print(disease_name)

    tk.messagebox.showinfo('Test Image Prediction',disease_name)
    Show_Bar_Chart(predictions)

def Show_Bar_Chart(predictions_list):
    objects=('No_DR','Mild','Moderate','Severe','Proliferate_DR')
    y_pos=np.arange(len(objects))

    flatten_prediction_list=list(chain.from_iterable(predictions_list))
    flatten_prediction_percentage=map(lambda x: round(x*100,2),flatten_prediction_list)

    performance=list(flatten_prediction_percentage)
    print(performance)
    plt.bar(y_pos,performance,align='center',alpha=0.5)

    for index,data in enumerate(performance):
        plt.text(x=index,y=data+1,s=f"{data}%",fontdict=dict(fontsize=10))

    plt.xticks(y_pos,objects)
    plt.ylabel('Percentage Probabilities')
    plt.title('Disease Prediction')

    plt.show()

if __name__=='__main__':
    model=load_training_model()
    root=tk.Tk()
    root.withdraw()

    MsgBox=tk.messagebox.askquestion('Tensorflow Predictions','Do you want to test images for predictions?')
    while MsgBox =='yes':
        MsgBox=tk.messagebox.askquestion('Test Image','Do you want to test new image')
        if MsgBox=='yes':
            predict_test_image_file(model)
        else:
            tk.messagebox.showinfo('Exit Application','Thank you for using the application')
            break