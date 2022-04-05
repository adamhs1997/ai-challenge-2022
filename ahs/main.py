import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

with open("trainData.csv", 'r') as f:
    train = np.array([list(map(int, next(f).strip().split(","))) for _ in range(10000)])

with open("trainTruth.csv", 'r') as f:
    labels = list(map(int, [next(f).strip() for _ in range(10000)]))

# Split our data
raw_train_data, raw_test_data, train_labels, test_labels = train_test_split(train,
                                                                            labels,
                                                                            test_size=0.33,
                                                                            random_state=42)


def classify(classifier, columns=None):
    if columns is not None:
        train_data = raw_train_data[:, columns]
        test_data = raw_test_data[:, columns]
    else:
        train_data = raw_train_data
        test_data = raw_test_data

    # Train our classifier
    classifier.fit(train_data, train_labels)

    # Make predictions
    preds = classifier.predict(test_data)

    # Evaluate accuracy
    print(type(classifier), accuracy_score(test_labels, preds))


def select_best_percent(selection_method, percent):
    selector = SelectPercentile(selection_method, percentile=percent)
    selector.fit(raw_train_data, train_labels)
    return selector.get_support(indices=True)


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


classify(GaussianNB())
classify(ComplementNB())
classify(LinearSVC())
classify(RandomForestClassifier())
classify(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))

"""
Without feature selection:
Gaussian: 6%
Complement: 20%
Linear: 12%
RFC: 40%
MLP: 4%
"""

# # Dim red--get top 25% of features by X2 test
# top_25 = select_best_percent(chi2, 25)
#
# classify(GaussianNB(), top_25)
# classify(ComplementNB(), top_25)
# classify(LinearSVC(), top_25)
# classify(RandomForestClassifier(), top_25)
# classify(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), top_25)

"""
Without top 25% chi2 feature selection:
Gaussian: 7%
Complement: 9%
Linear: 7%
RFC: 35%
MLP: 4%
"""

# # Dim red--get top 10% of features by X2 test
# top_10 = select_best_percent(chi2, 10)
#
# classify(GaussianNB(), top_10)
# classify(ComplementNB(), top_10)
# classify(LinearSVC(), top_10)
# classify(RandomForestClassifier(), top_10)
# classify(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), top_10)


"""
Without top 10% chi2 feature selection:
Gaussian: 6%
Complement: 6%
Linear: 5%
RFC: 19%
MLP: 4%
"""

# # Dim red--get top 5% of features by X2 test
# top_5 = select_best_percent(chi2, 5)
#
# classify(GaussianNB(), top_5)
# classify(ComplementNB(), top_5)
# classify(LinearSVC(), top_5)
# classify(RandomForestClassifier(), top_5)
# classify(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), top_5)


"""
Top 5% chi2 feature selection:
Gaussian: 6%
Complement: 7%
Linear: 5%
RFC: 14%
MLP: 4%
"""

# PCA 100

# classify_with_pca(GaussianNB(), 100)
# # classify_with_pca(ComplementNB(), 100)
# classify_with_pca(LinearSVC(), 100)
# classify_with_pca(RandomForestClassifier(), 100)
# classify_with_pca(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), 100)


"""
100 components feature selection:
Gaussian: 43%
Complement: NA
Linear: 8%
RFC: 79%
MLP: 4%
"""

# PCA 50
#
# classify_with_pca(GaussianNB(), 50)
# # classify_with_pca(ComplementNB(), 50)
# classify_with_pca(LinearSVC(), 50)
# classify_with_pca(RandomForestClassifier(), 50)
# classify_with_pca(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), 50)


"""
50 components feature selection:
Gaussian: 42%
Complement: NA
Linear: 8%
RFC: 81%
MLP: 4%
"""

# PCA 20

# classify_with_pca(GaussianNB(), 20)
# # classify_with_pca(ComplementNB(), 20)
# classify_with_pca(LinearSVC(), 20)
# classify_with_pca(RandomForestClassifier(), 20)
# classify_with_pca(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), 20)


"""
20 components feature selection:
Gaussian: 31%
Complement: NA
Linear: 5%
RFC: 75%
MLP 5%
"""


# Dim red--get top 25% of features by F-score test
# top_25_f = select_best_percent(f_classif, 25)
#
# classify(GaussianNB(), top_25_f)
# classify(ComplementNB(), top_25_f)
# classify(LinearSVC(), top_25_f)
# classify(RandomForestClassifier(), top_25_f)
# classify(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), top_25_f)

"""
Top 25% Fscore feature selection:
Gaussian: 7%
Complement: 10%
Linear: 7%
RFC: 35%
MLP: 4%
"""

# Dim red--get top 10% of features by Fscore test
# top_10_f = select_best_percent(f_classif, 10)
#
# classify(GaussianNB(), top_10_f)
# classify(ComplementNB(), top_10_f)
# classify(LinearSVC(), top_10_f)
# classify(RandomForestClassifier(), top_10_f)
# classify(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), top_10_f)

"""
Without top 10% Fscore feature selection:
Gaussian: 7%
Complement: 6%
Linear: 6%
RFC: 19%
MLP: 4%
"""

# Dim red--get top 5% of features by X2 test
top_5_f = select_best_percent(f_classif, 5)

classify(GaussianNB(), top_5_f)
classify(ComplementNB(), top_5_f)
classify(LinearSVC(), top_5_f)
classify(RandomForestClassifier(), top_5_f)
classify(MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), top_5_f)

"""
Without top 5% Fscore feature selection:
Gaussian: 6%
Complement: 7%
Linear: 6%
RFC: 14%
MLP: 4%
"""
