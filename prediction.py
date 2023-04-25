import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import img_to_array
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

from itertools import chain

from keras.models import model_from_json
from keras.models import load_model


Disease_classes={
    0:'No_DR',
    1:'Mild',
    2:'Moderate',
    3:'Severe',
    4:'Proliferate_DR'
}

def load_training_model():
    json_file = open('model_num.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # load weights into new model
    model.load_weights("model_num.h5")
    print("Loaded model from disk")
    
    model.save('model_num.hdf5')
    model=load_model('model_num.hdf5')
    return model

def predict_test_image_file(model):
    root=tk.Tk()
    root.withdraw()
    imageff=filedialog.askopenfilename()

    image=cv2.imread(imageff)

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
        plt.text(x=index,y=data+1,s=f"{data}",fontdict=dict(fontsize=10))

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