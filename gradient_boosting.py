from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


class GradientBoosting:
    _initialized = False
    _trained = False

    def __init__(self):
        self.params = {'n_estimators': 550, 'max_depth': 3,
                       'random_state': 0, 'learning_rate': 0.01}
        self.model = GradientBoostingClassifier(**self.params)
        self._initialized = True
        print('Gradient Boosting Classifier initialized with parameter:')
        print(self.params)
        print('\n')

    def train(self, x_train, y_train):
        print('\n')
        if self._initialized:
            self.model = self.model.fit(x_train, y_train)
            self._trained = True
            print('Model training complete.')
        else:
            print('Model not initialized. Training failed.')

    def getModel(self):
        if self._initialized:
            return self.model
        return None

    def classify(self, x_test, y_test):
        print('\n')
        if self._trained:
            output = self.model.predict(x_test)
            acc = accuracy_score(y_test, output)
            print('acc: %.4f %%' % (100 * acc))
