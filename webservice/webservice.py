import sys
from flask import Flask, jsonify, request, make_response, abort
import os
import nltk
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import time
import logging
import pickle
import re
import json

sys.path.append(".")
sys.path.append("..")
sys.path.append("webservice/models")

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
__location__ = os.path.realpath(os.path.join(
    os.getcwd(), os.path.dirname(__file__), 'models'))

# Loading models
model_impact = pickle.load(
    open(
        os.path.join(__location__, "impact.model"), "rb"
    )
)
model_ticket_type = pickle.load(
    open(
        os.path.join(__location__, "ticket_type.model"), "rb"
    )
)
model_category = pickle.load(
    open(
        os.path.join(__location__, "category.model"), "rb"
    )
)

model_urgency = pickle.load(
    open(
        os.path.join(__location__, "urgency.model"), "rb"
    )
)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/api/predictall', methods=['POST'])
def predictall():
    ts = time.gmtime()
    logging.info("Request received - %s" % time.strftime("%Y-%m-%d %H:%M:%S", ts))
    json_leido = json.loads(request.data)
    if (not json_leido) or ('description' not in json_leido):
        abort(400)
    description = json_leido['description']

    predicted_ticket_type = model_ticket_type.predict([description])[0]
    print("predicted ticket_type: "+str(predicted_ticket_type))

    predicted_category = model_category.predict([description])[0]
    print("predicted category: "+str(predicted_category))

    predicted_impact = model_impact.predict([description])[0]
    print("predicted impact: "+str(predicted_impact))

    predicted_urgency = model_urgency.predict([description])[0]
    print("predicted urgency: "+str(predicted_urgency))

    ts = time.gmtime()
    logging.info(
        "Request sent to evaluation - %s"
        % time.strftime("%Y-%m-%d %H:%M:%S", ts)
    )
    return jsonify({
        "description": description,
        "ticket_type": predicted_ticket_type,
        "category": predicted_category,
        "impact": predicted_impact,
        "urgency": predicted_urgency
    })

# exponer el modelo como servicio web
if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)

