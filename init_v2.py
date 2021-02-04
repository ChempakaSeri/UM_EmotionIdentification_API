from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from sklearn.externals import joblib
import xgboost as xgb

import pickle
import pandas as pd

#   Get headers of the statistical-based features
header = ['total_tweet','afraid_percent','anger_percent','bored_percent','excited_percent','happy_percent','relax_percent','sad_percent','avg_length','avg_ari','avg_char','std_dev','afraid_prob','anger_prob','bored_prob','excited_prob','happy_prob','relax_prob','sad_prob']

retName = ['Probability_afraid','Probability_anger','Probability_bored','Probability_excited','Probability_happy', 'Probability_relax', 'Probability_sad']

#   Use pickle to load in the pre-trained model

app = Flask (__name__)
CORS(app)

@app.route('/api/v1/boost', methods=['POST'])
def predict():
    # from sklearn.externals import joblib

    # joblib_file = '/Users/chempakaseri/Spyder/xgboost/Model/bst_model.pkl'
    # joblib_model = joblib.load(open(joblib_file, 'rb'))
    joblib_model = xgb.Booster({'nthread':4})
    # joblib_model.load_model('/Users/chempakaseri/Spyder/xgboost/Model/bst_model_19.pkl')

    joblib_model.load_model('/Users/chempakaseri/Spyder/xgboost/1022/bst_model_19.pkl')

    return_dict = {}
    
    features = request.json['data']
    print (request.json, flush=True)
    feature = [float(i) for i in features.split(',')]

    

    return_dict = {}

    values = pd.DataFrame( [feature],
                            columns=header,
                            dtype=float,
                            index=['input']
                            )

    input_variable = xgb.DMatrix(values)

    afraid_probability = joblib_model.predict(input_variable)[0][0]
    anger_probability = joblib_model.predict(input_variable)[0][1]
    bored_probability = joblib_model.predict(input_variable)[0][2]
    excited_probability = joblib_model.predict(input_variable)[0][3]
    happy_probability = joblib_model.predict(input_variable)[0][4]
    relax_probability = joblib_model.predict(input_variable)[0][5]
    sad_probability = joblib_model.predict(input_variable)[0][6]


    return_dict.update({ #word2seq_cnn, word2vec_cnn, ...
        retName[0]:str(afraid_probability), 
        retName[1]:str(anger_probability),
        retName[2]:str(bored_probability),
        retName[3]:str(excited_probability),
        retName[4]:str(happy_probability),
        retName[5]:str(relax_probability),
        retName[6]:str(sad_probability)
    })

    print (return_dict,flush=True)
    return(jsonify(return_dict))

def main():
    #   Load model into a dictionary
    app.run(debug=True, host='localhost', port=5050)   
#   running REST interface, port=6000 for direct test
if __name__ == "__main__":
    main()   