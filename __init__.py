import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from individual_company_stock import getHistoryData
import math
from flask import Flask, request, render_template, send_file
from Sentimental_Analysis import sentimental_analysis
from datetime import date, timedelta
import matplotlib
import json
import random
# import concurrent.futures
# import multiprocessing

matplotlib.use("agg")
app = Flask(__name__)

parent_dir = "D:\\programs\\ads_flask\\logres_op\\"
model_dir = os.path.join(parent_dir, "model\\")
app.config["UPLOAD_FOLDER"] = os.path.join(parent_dir, "temp_img")
data_dir="D:\\programs\\ads_flask\\logres_op\\app_hist_data"

data_files = os.listdir(model_dir)
cdate = date.today()
stock_list = []
models = []
hist_data = dict()
with open("D:\\programs\\ads_flask\\logres_op\\metrics.txt",'r') as json_file:
   metrics = json.load(json_file)

def strdate(cdate, daydiff=0):
    """returns datetime in string with the option to get a different date"""
    cdate = cdate - timedelta(days=daydiff)
    return cdate.strftime("%d-%m-%Y")


def get_his(stock, startdate, enddate):
    """retrives history for a stock for a given time period"""
    temp=[]
    f = 0
    while f < 10:
        try:
            print("retrieving data for stock:"+str(stock))
            temp = getHistoryData(stock, from_date=startdate, to_date=enddate)
            temp = temp.replace({",": ""}, regex=True)
            print("retrieved data for stock:"+str(stock))
            break
        except:
            print("failed"+str(f)+" times for"+str(stock)+"trying again")
            f = f + 1
            continue
    return temp


def load_stock_model(stock_name):
    """loads a particular model"""
    model_path = os.path.join(model_dir, stock_name + ".h5")
    # print("loading model:"+stock_name)
    model = load_model(model_path)
    print("loaded model for "+str(stock_name))
    return model

f1=0
f2=0
for data_file in data_files:
    stock_list.append(os.path.splitext(data_file)[0])
def pfn1():
    for i in range(len(stock_list)):
        models.append(load_stock_model(stock_list[i]))
def pfn2():
    ls_d=[]
    files=[]
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            ls_d.append(os.path.join(root, name))
    for i in range(len(files)):
        files[i]=files[i].replace(".csv","")
        f1=i
    for i in range(0,len(stock_list)):
        hist_data[stock_list[i]] = get_his(
            stock_list[i], strdate(cdate, daydiff=180), strdate(cdate)
        )
        f2=i
            
# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     executor=concurrent.futures.ProcessPoolExecutor(max_workers=2)
#     t1=executor.submit(pfn1)
#     t2=executor.submit(pfn2)
#     concurrent.futures.wait([t1,t2])
#     for completed_task in concurrent.futures.as_completed([t1, t2]):
#         pass
pfn1()
pfn2()

val = {}
prl= {}
print(stock_list)

def make_prediction(stock_name,i):
    """loads model,scrapes data, predicts 90 day future for a stock based on current day and returns the predictions"""
    # model=load_stock_model(stock_name)
    print(stock_name)
    input_data = hist_data[stock_name]
    print("before process")
    print(input_data[:10])
    input_data = pd.DataFrame(
        pd.to_numeric(pd.Series(input_data["close "].tail(91))), columns=["close "]
    )
    input_data = input_data.reset_index(drop=True)
    input_data["logret"] = np.log(input_data["close "]) - np.log(
        input_data["close "].shift(1)
    )
    input_data = input_data.drop(0, axis=0)
    input_data.reset_index(drop=True, inplace=True)
    val[stock_name]=(input_data["close "][0])
    print("after process")
    hist_data[stock_name]=input_data
    print(input_data.head())
    predictions = models[i].predict(
        np.array(input_data["logret"][input_data["logret"].size - 90 :]).reshape(
            1, 90, 1
        ),
        verbose=0
    )
    print("predictions initially")
    prl[stock_name]=predictions[0][89]
    print(predictions[:10])
    t = []
    for j in range(predictions.shape[0]):
        if j == 0:
            t.append(input_data["close "][0] * (math.e ** predictions[j]))
        else:
            t.append((math.e ** predictions[0][j]) * t[j - 1])
    predictions = pd.DataFrame(t,index=["close price"]).T
    
    print(str(stock_name)+" predictions")
    print(predictions[:10])
    return predictions  # future 90 day predictions from today for stock [stock_name]


def create_plot(predictions, name):
    """generates a plot which can be used in a webpage based on predictions.
    Use in conjunction with outputs from make_prediction"""
    fig = plt.figure(figsize=(15, 5))
    plt.scatter(np.arange(90), predictions, c="g", figure=fig,label="predicted values")
    plt.scatter(-1,val[name],c="r",label="current value",figure=fig)
    plt.plot(np.arange(90), predictions, figure=fig)
    plt.xticks(ticks=np.arange(0, 90, 5), labels=np.arange(1, 91, 5), figure=fig)
    plt.xlabel("created on " + str(date.today()), figure=fig)
    plt.ylabel("predicted value", figure=fig)
    plt.title("future 90-Day predictions for " + str(name), figure=fig)
    # plt.legend(figure=fig)
    return fig  # Return the figure object


s_pred = dict()
for i in range(len(stock_list)):
    s_pred[stock_list[i]] = make_prediction(stock_list[i],i)
print(s_pred)

def get_sentiment():
    print("val")
    print(val)
    temp = dict()
    for i in range(len(stock_list)):
        temp[stock_list[i]]=(val[stock_list[i]],(val[stock_list[i]]-prl[stock_list[i]])/val[stock_list[i]])
    op = sentimental_analysis(temp, 1000000)
    print(temp)
    print(op)
    return op

sent = get_sentiment()

def get_index(arr,val):
    for i in range(len(arr)):
        if(arr[i]==val):
            return i
    return 0
def table_dict(sentimen):
    res=[]
    for i in sentimen.keys():
        res.append([i,sentimen[i]])
    res=pd.DataFrame(res,columns=["stock","sentiment"])
    return res
res=table_dict(sent)
print(metrics)
@app.route("/")
def index():
    return render_template("index1.html", sentiment=res.to_html(),stockl=stock_list)

@app.route("/predict/")
def predict():
    stock_name = request.args.get(
        "stock_name"
    )  # Get the value of the "stock_name" query parameter
    if stock_name in stock_list:
        try:
            os.remove(os.path.join(app.config["UPLOAD_FOLDER"], "image.png"))
        except:
            print("no file")
        # p=make_prediction(stock_name)
        fig = create_plot(s_pred[stock_name], stock_name)
        # Save the figure to a file
        image_path = os.path.join(parent_dir, "temp_img", "image.png")
        fig.savefig(os.path.join(app.config["UPLOAD_FOLDER"], "image.png"))
        plt.close(fig)  # Close the figure to release memory
        if stock_name not in sent:
            return render_template(
                "index.html",
                stock_name=stock_name,
                value=val[stock_name],
                sentiment=random.choice(["Negative", "Positive"]),
                mae=metrics[stock_name][0],
                rmse=metrics[stock_name][1],
                data=pd.DataFrame(
                    np.array(s_pred[stock_name]).reshape((90, 1)), columns=["close price"]
                ).to_html(),
            )
        return render_template(
            "index.html",
            stock_name=stock_name,
            value=val[stock_name],
            sentiment=sent[stock_name],
            data=pd.DataFrame(
                np.array(s_pred[stock_name]).reshape((90, 1)), columns=["close price"]
            ).to_html(),
        )
    else:
        return "Model not found for the specified stock."


if __name__ == "__main__":
    app.run(debug=True)