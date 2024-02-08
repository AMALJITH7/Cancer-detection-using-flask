from flask import Flask
from flask import render_template,request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


app = Flask(__name__)

data=pd.read_csv('breast_cancer_data.csv')

x=data[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']]
y=data['diagnosis']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

lg=LogisticRegression()
lg.fit(x_train,y_train)


@app.route('/')
def first():
    return render_template('index.html')
    
    
@app.route('/out',methods=['GET','POST'])
def output():
    mean_radius=request.form['mean_radius']
    mean_texture=request.form['mean_texture']
    mean_perimeter=request.form['mean_perimeter']
    mean_area=request.form['mean_area']
    mean_smoothness=request.form['mean_smoothness']
    
    re=np.array([mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness])
    result= re.astype(np.float64)
    prediction=lg.predict([result])
    
    return render_template('prediction.html',ans=prediction)
    
    
    

if __name__ == '__main__':
    app.run(debug=True)    
