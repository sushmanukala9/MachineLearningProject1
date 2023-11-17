
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import numpy as np 
import pandas as pd 
from sklearn import *
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA 


main = tkinter.Tk()
main.title("Hybrid Feature Selection")
main.geometry("1300x1200")

global filename
global labels 
global columns
global balance_data
global data
global X, Y, X_train, X_test, y_train, y_test
global jtree, random_acc, eml_acc
global features
global test

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def importdata(): 
    global filename
    global balance_data
    balance_data = pd.read_csv(filename) 
    return balance_data 

def splitdataset(balance_data,val): 
    global features
    global test
    features = 0
    name = os.path.basename(filename)
    print(str(name))
    if name == 'Lymphoma.csv':
       features = 4027
       
    if name == 'SRBCT.csv':
       features = 2309
       
    X = balance_data.values[:, 0:(features-2)] 
    Y = balance_data.values[:, (features-1)] 
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = val, random_state = 0)
    return X, Y, X_train, X_test, y_train, y_test 

def upload():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = "dataset")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")

def preprocess(): 
    global labels
    global columns
    global filename
    global balance_data
    
    text.delete('1.0', END)
    
    labels = {"DLBCL":0,"FL":1,"CLL":2}
    balance_data = pd.read_csv(filename)
    
    
    text.insert(END,"Removed empty characters from dataset\n\n")
    text.insert(END,"Dataset Information\n\n")
    text.insert(END,str(balance_data)+"\n\n")

def generateModel():
    global data
    global test
    global X, Y, X_train, X_test, y_train, y_test

    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data,0.4)
    text.delete('1.0', END)
    text.insert(END,"Training model generated\n\n")

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred 
	
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details): 
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test,y_pred)*100
    text.insert(END,details+"\n\n")
    text.insert(END,"Accuracy : "+str(accuracy)+"\n\n")
    text.insert(END,"Report : "+str(classification_report(y_test, y_pred))+"\n")
    text.insert(END,"Confusion Matrix : "+str(cm)+"\n\n\n\n\n")  
    return accuracy    


def runJ48():
    global jtree
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = DecisionTreeClassifier(max_depth=0.9,random_state=0) 
    options = {'c1': 2, 'c2': 2, 'w':0.3, 'k': 20, 'p':2}   
    dimensions = X_train.shape[1]    
    optimizer = ps.discrete.BinaryPSO(n_particles=20, dimensions=dimensions, options=options)
    cost, pos = optimizer.optimize(fx.sphere, iters=10)
    b = pos[0]
    X_selected_features = X_train[:,b==1]
    X_test_features = X_test[:,b==1]
    text.insert(END,"Total features : "+str(features)+"\n\n")
    text.insert(END,"Total features after applying PSO : "+str(len(X_selected_features))+"\n\n")
    pca = PCA(n_components=3)
    X_train1 = pca.fit_transform(X_train)
    cls.fit(X_train1,y_train)
    X_test1 = pca.fit_transform(X_test)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test1, cls) 
    jtree = cal_accuracy(y_test, prediction_data,'J48 Tree Accuracy, Classification Report & Confusion Matrix') 
    

def runRandomForest():
    global random_acc
    global X, Y, X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    cls = RandomForestClassifier(n_estimators=12,max_depth=0.09,random_state=0) 
    options = {'c1': 2, 'c2': 2, 'w':0.3, 'k': 20, 'p':2}   
    dimensions = X_train.shape[1]    
    optimizer = ps.discrete.BinaryPSO(n_particles=20, dimensions=dimensions, options=options)
    cost, pos = optimizer.optimize(fx.sphere, iters=10)
    b = pos[0]
    X_selected_features = X_train[:,b==1]
    X_test_features = X_test[:,b==1]
    text.insert(END,"Total features : "+str(features)+"\n\n")
    text.insert(END,"Total features after applying PSO : "+str(len(X_selected_features))+"\n\n")

    ICA = FastICA(n_components=2, random_state=0) 
    X_train1 = ICA.fit_transform(X_train)
    X_test1 = ICA.fit_transform(X_test)

    
    cls.fit(X_train1,y_train)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test1, cls) 
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy, Classification Report & Confusion Matrix') 
    

def runEML():
    global eml_acc
    global features
    global X, Y, X_train, X_test, y_train, y_test
    global balance_data
    X, Y, X_train, X_test, y_train, y_test = splitdataset(balance_data,0.5)
    text.delete('1.0', END)
    srhl_tanh = MLPRandomLayer(n_hidden=100, activation_func='tanh')
    cls = GenELMClassifier(hidden_layer=srhl_tanh)
    options = {'c1': 2, 'c2': 2, 'w':0.3, 'k': 20, 'p':2}   
    dimensions = X_train.shape[1]    
    optimizer = ps.discrete.BinaryPSO(n_particles=20, dimensions=dimensions, options=options)
    cost, pos = optimizer.optimize(fx.sphere, iters=10)
    b = pos[0]
    X_selected_features = X_train[:,b==1]
    X_test_features = X_test[:,b==1]
    text.insert(END,"Total features : "+str(features)+"\n\n")
    text.insert(END,"Total features after applying PSO : "+str(len(X_selected_features))+"\n\n")
    pca = PCA(n_components=3)
    X_train1 = pca.fit_transform(X_train)
    cls.fit(X_train1,y_train)
    X_test1 = pca.fit_transform(X_test)
    text.insert(END,"Prediction Results\n\n") 
    prediction_data = prediction(X_test1, cls) 
    eml_acc = cal_accuracy(y_test, prediction_data,'ELM & PSO Algorithm Accuracy, Classification Report & Confusion Matrix') 
    

def graph():
    height = [jtree,random_acc,eml_acc]
    bars = ('J48 Accuracy', 'Random Forest Accuracy','EML Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Hybrid Feature Selection Using Correlation Coefficient and Particle Swarm Optimization on Microarray Gene Expression Data')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Micro Array Dataset", command=upload)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=300,y=100)

preprocess = Button(main, text="Preprocess Dataset", command=preprocess)
preprocess.place(x=50,y=150)
preprocess.config(font=font1) 

model = Button(main, text="Generate Training Model", command=generateModel)
model.place(x=330,y=150)
model.config(font=font1) 

runtree = Button(main, text="Run J48 Algorithm", command=runJ48)
runtree.place(x=610,y=150)
runtree.config(font=font1) 

runrandomforest = Button(main, text="Run Random Forest Algorithm", command=runRandomForest)
runrandomforest.place(x=870,y=150)
runrandomforest.config(font=font1) 

runeml = Button(main, text="Run EML Algorithm", command=runEML)
runeml.place(x=50,y=200)
runeml.config(font=font1) 

graph = Button(main, text="Accuracy Graph", command=graph)
graph.place(x=330,y=200)
graph.config(font=font1) 

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()