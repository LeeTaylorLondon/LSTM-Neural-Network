# LSTM-Neural-Network

## Description
This project features, data padding, data preprocessing, and a sequential neural network. 
Credit to the book Neural Network Projects with Python by James Loy. The file main.py contains all the functions
in the order executed. The data engineering and model structure can be viewed in main.py. The models that are
saved and can be loaded predict the sentiment of movie reviews to an 80% accuracy.

## Installation
* Pip install tensorflow (built with 2.5.0)

## Usage
The main.py file contains three sections. Data exploration, model building and training, and lastly 
model loading and testing. The first two sections are commented out. Therefore upon running the main.py file 
some models are loaded and then tested, with their accuracy outputted. You may uncomment multiple sections at
once to execute and view the other components of the program.

To get better usage out of this project the functions should be read and understood in the order executed. 
Alongside the book Neural Network Projects with Python by James Loy. 

## Neural Network Details
Input layer, units=10,000
LSTM layer 1, units=128, activation=tanh 
Output layer, units=1, activation=sigmoid
Optimizer Adam, RMSprop, SGD, Loss binary_crossentropy

SGD Testing: 50%, Adam: ~80%, RMSprop: ~80%

## Credits
* Author: James Loy
* Modified & Studied by: Lee Taylor
