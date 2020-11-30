from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
data = pd.read_csv('modified.csv')
pipe = pickle.load(open("LaptopPredictorModel.pkl",'rb'))

@app.route('/')
def index():

    companies = sorted(data['Company'].unique())
    typename = sorted(data['TypeName'].unique())
    screen_size = sorted(data['Inches'].unique())
    screen_resolution = sorted(data['ScreenResolution'].unique())
    cpu = sorted(data['Cpu'].unique())
    ram = sorted(data['Ram'].unique())
    memory = sorted(data['Memory'].unique())
    gpu = sorted(data['Gpu'].unique())

    company_wise = dict()
    for company in companies:
        company_dict = dict()
        company_dict['model'] =sorted(data[data['Company']==company]['Product'].unique())
        company_dict['opsys'] = sorted(data[data['Company']==company]['OpSys'].unique())
        company_dict['weight'] = sorted(data[data['Company'] == company]['Weight'].unique())
        company_wise[company] = company_dict

    companies.insert(0,'Select Company')
    company_dict = dict()
    company_dict['model'] = []
    company_dict['opsys'] = []
    company_dict['weight'] = []

    company_wise['Select Company'] = company_dict
    #print(companies,typename,screen_size,screen_resolution,cpu,ram,memory, gpu, company_wise)


    return render_template('index.html', companies=companies, typename=typename, screen_size=screen_size,
                           screen_resolution=screen_resolution, cpu=cpu, ram=ram, memory=memory, gpu=gpu,
                           company_wise=company_wise)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get("company")
    model = request.form.get("model")
    typename = request.form.get("typename")
    screen_size = float(request.form.get("screen_size"))
    screen_resolution = request.form.get("screen_resolution")
    cpu = request.form.get("cpu")
    ram = request.form.get("ram")
    memory = request.form.get("memory")
    gpu = request.form.get("gpu")
    opsys = request.form.get("opsys")
    weight =  float(request.form.get("weight"))

    print(company,model,typename,screen_size,screen_resolution,cpu,ram,memory,gpu,opsys,weight)

    prediction = pipe.predict(pd.DataFrame([[company,model,typename,screen_size,screen_resolution,cpu,ram,memory,gpu, opsys,weight]],
                 columns=['Company','Product','TypeName','Inches','ScreenResolution','Cpu','Ram','Memory','Gpu','OpSys','Weight']))

    return str(np.round(prediction[0],2))


if __name__=="__main__":
    app.run(debug=True)