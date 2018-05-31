import pandas as pd
import numpy
from sklearn.preprocessing import LabelEncoder
from random_forest import RandomForest
from gradient_boosting import GradientBoosting
from sklearn.model_selection import cross_val_score, ShuffleSplit

numpy.random.seed(7)
number = LabelEncoder()


# This method perofrms n fold cross validation on the model
# It uses f1 as the scoring function
def testClassifier(model, train, target, cv=5):
    if model is not None:
        cv = ShuffleSplit(n_splits=cv, test_size=0.2, random_state=0)
        score = cross_val_score(model,
                                train,
                                target,
                                cv=cv,
                                scoring='f1_macro')
        print ("cross val acc: %.4f %%" % (100 * score.mean()))

def main():
    # Imputed data set
    train = pd.read_csv('./data/imputed_complete_data.csv')
    # Non imputed data set
    # train_complete = pd.read_csv('./data/complete_data.csv')

    # List of features that were used
    features = ['signup_city_name', 'event_name', 'signup_sub_channel',
                'driver_flow', 'device_os', 'signup_mega_region',
                'signup_country_name', 'signup_channel', 'email_domain',
                'is_polymorphed', 'age', 'gender',  'os_category',
                'launch_date_ms', 'source_description', 'datestr']

    # Handling categorical columns in data
    for feature in features:
        train[feature] = number.fit_transform(train[feature].astype('str'))

    # Extract features and target
    x_features = train[list(features)].values
    x_target = train['normalized_score'].values

    # Run cross validation on gradient boosting
    model = GradientBoosting()
    testClassifier(model.getModel(), x_features, x_target, 10)

    # Run cross validation on random forest
    model = RandomForest()
    testClassifier(model.getModel(), x_features, x_target, 10)


if __name__ == '__main__':
    main()
