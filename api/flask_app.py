import torch
import pickle
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import traceback
import logging
from pytorch_lstm import ToxicLSTM
from utils import preprocess

app = Flask(__name__)
CORS(app)

MAX_LEN = 250
TOK_PATH = F"/home/TatianaG/mysite/tokenizer.pickle"
MODEL_PATH = F"/home/TatianaG/mysite/model.pt"

CATEGORIES = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']

with open(TOK_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

max_features = len(tokenizer.word_index)

state_dict = torch.load(MODEL_PATH, map_location='cpu')
model = ToxicLSTM(6, max_features, 300)
model.load_state_dict(state_dict['state_dict'])
model.eval()


def getPredictions(comments, model):

  try:

    comments = torch.tensor(comments, dtype=torch.long)
    out = model(comments)
    out = torch.sigmoid(out).tolist()[0]
    out = ["{:.2f}".format(y) for y in out]
    out = dict(zip(CATEGORIES, out))
    out['success'] = 'true'
    return out

  except Exception as e:
    logging.error(traceback.format_exc())
    out = {'success': 'false', 'reason': 'model_error'}
    return out


def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():

    if request.method == 'POST':

        comment = str(request.form['comment'])

        try:
            comment = preprocess(comment)

        except Exception as e:
            logging.error(traceback.format_exc())
            out = {'success': 'false', 'reason': 'preprocessing_error'}


        try:
            comment = tokenizer.texts_to_sequences([comment])[0]
            tmp = list(filter(lambda x: x != 1, comment))
            if not tmp:
               out = {'success': 'false', 'reason': 'not_enough_data'}
               return out
            comments = [comment]

        except Exception as e:
            logging.error(traceback.format_exc())
            out = {'success': 'false', 'reason': 'preparation_error'}
            return out

        result = getPredictions(comments, model)

        return jsonify(result)

    else:
        return jsonify({'toxicity': str(0)})


app.after_request(add_cors_headers)