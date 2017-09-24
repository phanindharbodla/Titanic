import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def main():
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    cat_vars = ['Pclass', 'Embarked', 'Sex', 'Deck']
    columns = ['Pclass', 'Age', 'Sex', 'Fare', 'SibSp', 'Parch', 'Embarked', 'family', 'Deck']
    target = np.array(train.Survived).transpose()

    # Filling missing data and labeling categorical columns
    train["Embarked"] = train["Embarked"].fillna("C")
    train["Cabin"] = train["Cabin"].fillna('H')
    train["Deck"] = train["Embarked"].str[0]
    train["Age"] = train["Age"].fillna(train.groupby(["Survived", "Sex", "Pclass"])["Age"].transform("median"))
    train["family"] = (train["SibSp"] + train["Parch"] + 1)
    test["Embarked"] = test["Embarked"].fillna("C")
    test.set_value(152, 'Fare', train['Fare'].median())
    test["Cabin"] = test["Cabin"].fillna('H')
    test["Deck"] = test["Cabin"].str[0]
    test["Age"] = test["Age"].fillna(test.groupby(["Sex", "Pclass"])["Age"].transform("median"))
    test["family"] = (test["SibSp"] + test["Parch"] + 1)

    labeler = preprocessing.LabelEncoder()
    for col in cat_vars:
        train[col] = labeler.fit_transform(train[col])
        test[col] = labeler.fit_transform(test[col])

    forest = RandomForestClassifier(max_depth=5, n_estimators=120, min_samples_split=2)
    my_forest = forest.fit(train[columns], target)
    predictions = my_forest.predict(train[columns])
    print(accuracy_score(predictions, target))
    print("( tp, fp, fn, tn ) = ", np.reshape(confusion_matrix(predictions, target), 4))
    test_predictions = my_forest.predict(test[columns])
    output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': test_predictions})
    output.to_csv("Submission.csv", index=False)


if __name__ == '__main__':
    main()
