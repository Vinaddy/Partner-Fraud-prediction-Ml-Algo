from keras.models import Sequential
from keras.layers import Dense
import numpy as np

np.random.seed(7)


class NeuralNet:
    _initialized = False
    _trained = False

    def __init__(self):
        # create model
        self.epochs = 25
        self.batch_size = 10
        self.model = Sequential()
        self._initialized = True

    def train(self, x_train, y_train):
        if self._initialized:
            inputNum = len(x_train[0])
            self.model.add(
                Dense(inputNum, input_dim=inputNum, activation='relu'))
            self.model.add(Dense((inputNum+1)/2, activation='relu'))
            self.model.add(Dense(1, activation='sigmoid'))
            # Compile model
            self.model.compile(loss='binary_crossentropy',
                               optimizer='adam', metrics=['accuracy'])
            self.model.fit(x_train, y_train, epochs=self.epochs,
                           batch_size=self.batch_size)
            self._trained = True
            print ('Model training complete.')
        else:
            print ('Model not initialized. Training failed.')

    def classify(self, x_test, y_test):
        if self._trained:
            scores = self.model.evaluate(x_test, y_test)
            print("\n%s: %.2f%%" %
                  (self.model.metrics_names[1], scores[1]*100))
