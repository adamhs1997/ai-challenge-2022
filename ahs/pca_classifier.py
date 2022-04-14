from datetime import datetime

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score

print("Start", datetime.now())

with open("trainData.csv", 'r') as f:
    train = np.array(list(map(lambda s: list(map(int, s.strip().split(","))), f.readlines())))

with open("trainTruth.csv", 'r') as f:
    labels = list(map(int, map(lambda s: s.strip(), f.readlines())))

raw_train_data, raw_test_data, train_labels, test_labels = train_test_split(train,
                                                                            labels,
                                                                            test_size=0.33,
                                                                            random_state=42)

print("Data split!", datetime.now())


def classify_with_pca(classifier, n_components):
    # https://stats.stackexchange.com/a/144447
    pca = PCA(n_components=n_components)
    train_data_transformed = pca.fit_transform(raw_train_data, train_labels)
    test_data_transformed = pca.transform(raw_test_data)

    classifier.fit(train_data_transformed, train_labels)

    # Make predictions
    preds = classifier.predict(test_data_transformed)

    # Evaluate accuracy
    print(type(classifier), accuracy_score(test_labels, preds))

    return preds


def cross_validate(classifier, n_components):
    pca = PCA(n_components=n_components)
    train_data_transformed = pca.fit_transform(raw_train_data, train_labels)

    cv = KFold(n_splits=10, random_state=0, shuffle=True)
    scores = cross_val_score(classifier, train_data_transformed, train_labels, scoring='accuracy', cv=cv, n_jobs=3)
    print('{} - cross-val done!'.format(datetime.now()))
    print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


# predictions = classify_with_pca(RandomForestClassifier(), 50)
# with open("predictions.csv", 'w') as outf:
#     for p in predictions:
#         print(p, file=outf)

cross_validate(RandomForestClassifier(), 50)
