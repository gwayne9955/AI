import tensorflow as tf
from numpy import exp, array, random, dot
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense

# The training set. We have 7 examples, each consisting of 3 input values
# and 1 output value.
training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

# Creating the model using the Keras Sequential API
model = Sequential()

# Input Layer
model.add(Dense(1, input_dim=3, activation="sigmoid"))

# Hidden Layer, 4 neurons
model.add(Dense(4, activation="sigmoid"))

# Output Layer
model.add(Dense(1, activation="sigmoid"))

# Configuring the learning process
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

# Training the machine with class based binary fitting, and sgd optimizer with
# our training input arrays and single output values
model.fit(training_set_inputs, training_set_outputs, epochs=10, batch_size=7)

test = array([[0, 1, 0]])
print("Considering brand new data", test)

# Calculating a prediction of a 0 or 1 binary classifier based on the input
prediction = model.predict_classes(test)

# !!! Should print 0, or whichever value (0 or 1) is the XOR of the first two
# values in the validating test array !!!
print("AI predicts this new data classifies as", prediction[0][0])

if test[0][0] != prediction[0][0]:
    print("The predicted classification of", prediction[0][0], "is correct")
else:
    print("The predicted classification of", prediction[0][0], "is incorrect")