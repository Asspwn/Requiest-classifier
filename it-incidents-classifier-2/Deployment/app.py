#!/usr/bin/env python
# coding: utf-8

# In[16]:

from flask_swagger_ui import get_swaggerui_blueprint
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

from flask_apispec import marshal_with
from flask_apispec.views import MethodResource
from marshmallow import Schema, fields


# In[17]:




def remove(text):
    return text[:]

def r_dup(sent):
    words = sent.split() 
    res = [] 
    for word in words: 
        # If condition is used to store unique string  
        # in another list 'k'  
        if (sent.count(word)>1 and (word not in res)or sent.count(word)==1): 
            res.append(word) 
    return ' '.join(res)

def text_process(mess):
    
    mess = mess.replace('/', ' ')
    mess = mess.replace('(',' ')
    mess = mess.replace(')',' ')
    mess = mess.replace('-',' ')
    
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    nopunc = [word for word in nopunc.split() if word.lower() not in stopwords.words('russian')]
    return ' '.join(nopunc)

def lemmatize(text):
    res = list()
    words = text.split() # разбиваем текст на слова
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.normal_form)
        
    #removing numbers    
    res = [''.join(x for x in i if x.isalpha()) for i in res] 
    while '' in res:
        
        res.remove('')     
    
    return ' '.join(res)

def pos_tag(text):
    words = text.split()
    res = list()
    for word in words:
        p = morph.parse(word)[0]
        res.append(p.tag.POS)
    return res      


# In[18]:
###################################################################



from flask import Flask
from flask_restful import Resource, Api
from apispec import APISpec
from marshmallow import Schema, fields
from apispec.ext.marshmallow import MarshmallowPlugin
from flask_apispec.extension import FlaskApiSpec
from flask_apispec.views import MethodResource
from flask_apispec import marshal_with, doc, use_kwargs


app = Flask(__name__)
api = Api(app)

model = pickle.load(open('model_LR.pkl', 'rb'))

app.config.update({
    'APISPEC_SPEC': APISpec(
        title='it-incident-classifier',
        version='v1',
        plugins=[MarshmallowPlugin()],
        openapi_version='2.0.0',
        description='IT Request Classifier backed with Machine Learning'
    ),
    'APISPEC_SWAGGER_URL': '/swagger/',  # URI to access API Doc JSON 
    'APISPEC_SWAGGER_UI_URL': '/swagger-ui/'  # URI to access UI of API Doc
})
docs = FlaskApiSpec(app)


#################################################################



class AwesomeResponseSchema(Schema):
    message = fields.Str(default='Success')
    
class AwesomeRequestSchema(Schema):
    api_type = fields.String(required=True, description="API type")
    
    

class AwesomeAPI(MethodResource, Resource):
    @doc(description='Enter Text.', tags=['Awesome'])
    @use_kwargs(AwesomeRequestSchema, location=('json'))
    @marshal_with(AwesomeResponseSchema)  # marshalling
    @app.route('/')
    def home(self, **kwargs):
        return render_template('index.html')

   
    @app.route('/predict',methods=['POST'])
    def predict():
        '''
        For rendering results on HTML GUI
        '''
        int_features = request.form.values()
        final_features = int_features
        prediction_proba = model.predict_proba(final_features)[0][1]
        output_proba = round(prediction_proba, 2)
        #output = round(prediction[0], 2)
        return render_template('index.html', prediction_text='Probability of incident request is {}'.format(output_proba))
        #return render_template('index.html', prediction_text='Prediction {}'.format(output))
    
    
    @doc(description='Enter Text.', tags=['Awesome'])
    @use_kwargs(AwesomeRequestSchema, location=('json'))
    @marshal_with(AwesomeResponseSchema)  # marshalling
    @app.route('/predict_api',methods=['POST'])
    def post(self, **kwargs):
        def predict_api():
            '''
            For direct API calls trought request
            '''
            res = request.get_json(force=True)
            data = json.loads(res.text)
            data = json.dumps(data)
            data = data.apply(r_dup)
            data = data.apply(text_process)
            data = data.apply(lemmatize)
            prediction_proba = model.predict_proba(data)[0][1]

            output = prediction_proba
            return jsonify(output)
            return {'message': 'My First Awesome API'}

api.add_resource(AwesomeAPI, '/', )
docs.register(AwesomeAPI) 

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)


# In[ ]:




