import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def main():
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    cat_vars = ['Pclass', 'Embarked', 'Sex', 'Deck']
    columns = ['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch', 'Embarked', 'family', 'Deck']
    target = np.array(train.Survived).transpose()
    train = train.join(train.Name.str.split(',', 1, expand=True).rename(columns={0: 'FamilyCode', 1: 'PassengerName'}))
    test = train.join(test.Name.str.split(',', 1, expand=True).rename(columns={0: 'FamilyCode', 1: 'PassengerName'}))

    # Filling missing data and labeling categorical columns
    train["Embarked"] = train["Embarked"].fillna("C")
    train["Deck"] = train["Cabin"].fillna('H').str[0]
    train["Age"] = train["Age"].fillna(train.groupby(["Survived", "Sex", "Pclass"])["Age"].transform("median"))
    train["family"] = (train["SibSp"] + train["Parch"] + 1)
    test.set_value(152, 'Fare', train[(train['Pclass'] == 3) & (train['Embarked'] == 'S')]['Fare'].median())
    test["Deck"] = test["Cabin"].fillna('H').str[0]
    test["Age"] = test["Age"].fillna(train.groupby(["Sex", "Pclass"])["Age"].transform("median"))
    test["family"] = (test["SibSp"] + test["Parch"] + 1)

    labeler = preprocessing.LabelEncoder()
    for col in cat_vars:
        train[col] = labeler.fit_transform(train[col])
        test[col] = labeler.fit_transform(test[col])

    forest = RandomForestClassifier(max_depth=5, n_estimators=120, min_samples_split=2)
    gbc = GradientBoostingClassifier(learning_rate=0.05)
    my_forest = forest.fit(train[columns], target)
    my_gbc = gbc.fit(train[columns], target)
    predictions = my_forest.predict(train[columns])
    predictions_gbc = my_gbc.predict(train[columns])
    print('Random Forest')
    print(accuracy_score(predictions, target))
    print("( tp, fp, fn, tn ) = ", np.reshape(confusion_matrix(predictions, target), 4))
    print(forest.feature_importances_)
    # train['predictions'] = predictions
    # train.to_csv("validation.csv", index=False)
    print('Gradient Boost')
    print(accuracy_score(predictions_gbc, target))
    print("( tp, fp, fn, tn ) = ", np.reshape(confusion_matrix(predictions_gbc, target), 4))
    print(gbc.feature_importances_)
    test_rf = my_forest.predict(test[columns])
    test_gb = my_gbc.predict(test[columns])
    pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test_rf}).to_csv("output_rf.csv", index=False)
    pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test_gb}).to_csv("output_gb.csv", index=False)


if __name__ == '__main__':
    main()
