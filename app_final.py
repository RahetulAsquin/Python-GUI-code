import warnings
warnings.filterwarnings("ignore")
import sys
from fbprophet import Prophet
#from tkinter import ttk
import pandas as pd
import pickle
import matplotlib.pyplot as plt
#from tkinter import *
from tkinter import messagebox
from pandastable import Table
from prettytable import PrettyTable
import PIL.Image
import PIL.ImageTk


#root = Tk(className=' BR1 Predictive Model')

#load the save model
with open("Prophet1.pkl",'rb') as mm:
  loaded=pickle.load(mm)

counter=0
def show_table(df):
    pt = Table(frame, dataframe=df)
    if counter>1:
        frame.pack_forget()
        pt.redraw()
        pt.show()
    else:
        pt.show()
def pred(f1):
    flag=0
    n=0
    if f1.isdigit()==True:
        flag=1
    else:
        flag=0
    if flag==1:
        f=int(f1)
        if f>0 :
            future = loaded.make_future_dataframe(periods=f,freq="MS")
            future1 =future[-f:]
            forecast = loaded.predict(future1)
            y_axis=forecast['ds']
            predicted_values=forecast['yhat']
            a=round(predicted_values)
            result = pd.concat([y_axis, a], axis=1)
            res=result[['ds','yhat']]
            res['ds']=res['ds'].dt.date
            res=res.rename({'ds': 'Date', 'yhat': 'Count of Cases'}, axis=1)
            show_table(res[['Date','Count of Cases']])

    

    
def predict_plot(f1):
    n=0
    if f1.isdigit()==True:
        n=1
    else:
        n=0
    if n==1:
        f=int(f1)
        if f>0:
            future = loaded.make_future_dataframe(periods=f,freq="MS")
            future1 =future[-f:]
            forecast = loaded.predict(future1)
            y_axis=forecast['ds']
            predicted_values=forecast['yhat']
            a=round(predicted_values)
            result = pd.concat([y_axis, a], axis=1)
            res=result[['ds','yhat']]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax1=plt.axes()
            ax.set_facecolor("lightgreen")
            plt.plot (res['ds'],res['yhat'],'r', lw = 0.75) 
            plt.title('Predicted Number of cases in upcoming Months') 
            plt.ylabel('No of cases')
            plt.xlabel('Date')
            plt.show()


