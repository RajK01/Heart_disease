from flask import Flask,request,render_template, url_for
import numpy as np
import pickle

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 7)
    loaded_model = pickle.load(open("heart_disease.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
 

@app.route('/result', methods = ['GET','POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        output = ValuePredictor(to_predict_list) 
        print(output)      
        if output == 1:
            prediction ='Risk of Heart Disease'
        else:
            prediction ='Do not have risk of Heart Disease'           
        return render_template("index.html", prediction = prediction)

if __name__=='__main__':
    app.run(debug=True)