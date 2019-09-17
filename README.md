# detoxify
Chrome extension which detects, classifies and highlights toxic comments under Youtube videos

There are three parts in the application:

1. [Modelling](notebooks/)

	* [_youtube_api.ipynb_](notebooks/youtube_api.ipynb) - shows how to use youtube API in python
	* [_toxic_lstm.ipynb_](notebooks/toxic_lstm.ipynb) - model training
	* [_inference.ipynb_](notebooks/inference.ipynb) - inference
	* [_bert.ipynb_](notebooks/bert.ipynb) - BERT training (work in progress)

The model is built using the dataset from recent [Jigsaw Kaggle competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). 

2. [Chrome Extension](chrome_extension/)

	* Go to chrome://extensions
	* Enable Developer mode by toggling the button in the upper-right corner
	* Use the 'Load unpacked extension' option and select the folder, containing the extension


3. [Flask App](flask_app/)
