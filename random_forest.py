from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class RandomForest:
    _initialized = False
    _trained = False

    def __init__(self):
        self.numOfTrees = 125
        self.params = {'n_estimators': self.numOfTrees}
        self.model = RandomForestClassifier(**self.params)
        self._initialized = True
        print('\nRandom Forest initialized with parameters:')
        print(self.params)
        print('\n')

    def getModel(self):
        if self._initialized:
            return self.model
        return None

    # This function trains the model on a training set
    def train(self, x_train, y_train):
        print('\n')
        if self._initialized:
            self.model = self.model.fit(x_train, y_train)
            self._trained = True
            print('Model training complete.')
        else:
            print('Model not initialized. Training failed.')

    def classify(self, x_test, y_test):
        print('\n')
        if self._trained:
            output = self.model.predict(x_test)
            acc = accuracy_score(y_test, output)
            print('acc: %.4f %%' % (100 * acc))
