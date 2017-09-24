import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if str.find(big_string, substring) != -1:
            return substrings.index(substring)
    print(big_string)
    return np.nan


def pred_eval(pred_vec, target):
    result = dict()
    cm = confusion_matrix(pred_vec, target)
    true_positive = cm[0][0]
    true_negative = cm[1][1]
    false_positive = cm[0][1]
    false_negative = cm[1][0]
    result['accurate'] = true_positive + true_negative
    result['incorrect'] = false_positive + false_negative
    result['accuracy'] = float(result['accurate']) / float(target.size)
    return result


def main():
    # reading data from files
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    #Filling missing data and assinging numbers to categorical columns
    train.loc[train["Sex"] == "male", "Sex"] = 0
    train.loc[train["Sex"] == "female", "Sex"] = 1
    train["Embarked"] = train["Embarked"].fillna("C")
    train.loc[train["Embarked"] == "S", "Embarked"] = 0
    train.loc[train["Embarked"] == "C", "Embarked"] = 1
    train.loc[train["Embarked"] == "Q", "Embarked"] = 2
    test.loc[test["Sex"] == "male", "Sex"] = 0
    test.loc[test["Sex"] == "female", "Sex"] = 1
    test["Embarked"] = test["Embarked"].fillna("S")
    test.loc[test["Embarked"] == "S", "Embarked"] = 0
    test.loc[test["Embarked"] == "C", "Embarked"] = 1
    test.loc[test["Embarked"] == "Q", "Embarked"] = 2
    test['Fare'][152] = train['Fare'].median()
    train["Cabin"] = train["Cabin"].fillna('H')#filling nulls with H
    test["Cabin"] = test["Cabin"].fillna('H')
    cabin_list = ['F G', 'A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'H']
    train['Deck'] = train['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    test['Deck'] = test['Cabin'].map(lambda x: substrings_in_string(x, cabin_list))
    train["family"] = (train["SibSp"] + train["Parch"] + 1)
    test["family"] = (test["SibSp"] + test["Parch"] + 1)
    train["Age"] = train["Age"].fillna(train.groupby(["Survived", "Sex", "Pclass"])["Age"].transform("median")) # filling missing age with relevant medians
    test["Age"] = test["Age"].fillna(test.groupby(["Sex", "Pclass"])["Age"].transform("median"))
    target = np.array(train.Survived).transpose()
    features_forest = np.array([train.Pclass, train.Age, train.Sex, train.Fare, train.SibSp,
                                train.Parch, train.Embarked, train.family, train.Deck]).transpose()
    test_features_forest = np.array([test.Pclass, test.Age, test.Sex, test.Fare, test.SibSp, test.Parch, test.Embarked,
                                     test.family, test.Deck]).transpose()
    forest = RandomForestClassifier(max_depth=5, n_estimators=120, min_samples_split=2)
    my_forest = forest.fit(features_forest, target)
    my_forest.score(features_forest, target)
    pred_vec_forest = my_forest.predict(features_forest)
    mat = pred_eval(pred_vec_forest, target)
    pred_forest = my_forest.predict(test_features_forest)
    print(mat)
    print(my_forest.feature_importances_)
    StackingSubmission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': pred_forest})
    StackingSubmission.to_csv("Submission.csv", index=False)


if __name__ == '__main__':
    main()