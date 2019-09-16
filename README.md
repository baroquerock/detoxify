# detoxify
Chrome extension which detects, classifies and highlights toxic comments under Youtube videos

There are three parts in the application:

1. [Modelling](notebooks/)

The model is built using the dataset from recent [Jigsaw Kaggle competition](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). 

2. [Chrome Extension](chrome_extension/)

	* Go to chrome://extensions
	* Enable Developer mode by toggling the button in the upper-right corner
	* Use the 'Load unpacked extension' option and select the folder, containing the extension


3. [Flask App](flask_app/)
