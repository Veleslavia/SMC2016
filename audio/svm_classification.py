import pandas as pd
import pickle
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from utils import visualization

# IRMAS - 30
THRESHOLD = 20

labels = None

def feature_preprocessing(datafile, dataset):

    # import some data to play with
    data = pd.DataFrame.from_csv(datafile)

    # delete underrepresented classes

    selective_data = data.groupby(['class'])['zcr.mean'].count()
    classes = selective_data[(selective_data > THRESHOLD)].index.values
    data = data.loc[data['class'].isin(classes.tolist())]

    data = data.dropna(axis=1, how='any')

    y = data['class']
    X = data.drop(['class'], axis=1).values
    X = preprocessing.Imputer().fit_transform(X)

    if dataset == 'IRMAS':
        # IRMAS classes
        labels = ['cello', 'clarinet', 'flute', 'guitar (acoustic)',
                  'guitar (electric)', 'organ', 'piano', 'saxophone',
                  'trumpet', 'violin', 'voice']
    elif dataset == 'RWC':
        # RWC classes
        rwc_classes = {1: 'PIANOFORTE', 2: 'ELECTRIC PIANO ', 3: 'HARPSICHORD ', 4: 'GLOCKENSPIEL', 5: 'MARIMBA',
                       6: 'PIPE ORGAN', 7: 'ACCORDION ', 8: 'HARMONICA ', 9: 'CLASSIC GUITAR ', 10: 'UKULELE',
                       11: 'ACOUSTIC GUITAR ', 12: 'MANDOLIN', 13: 'ELECTRIC GUITAR', 14: 'ELECTRIC BASS ',
                       15: 'VIOLIN', 16: 'VIOLA', 17: 'CELLO', 18: 'CONTRABASS ', 19: 'HARP', 20: 'TIMPANI ',
                       21: 'TRUMPET', 22: 'TROMBONE', 23: 'TUBA', 24: 'HORN', 25: 'SOPRANO SAX', 26: 'ALTO SAX',
                       27: 'TENOR SAX', 28: 'BARITONE SAX', 29: 'ENGLISH HORN', 30: 'BASSOON ', 31: 'CLARINET',
                       32: 'PICCOLO', 33: 'FLUTE', 34: 'RECORDER ', 35: 'SHAKUHACHI ', 36: 'BANJO', 37: 'SHAMISEN ',
                       38: 'KOTO ', 39: 'SHO', 40: 'JAPANESE PERCUSSION', 41: 'CONCERT DRUMS ',  42: 'ROCK DRUMS ',
                       43: 'JAZZ DRUMS ', 44: 'PERCUSSION', 45: 'SOPRANO ',  46: 'ALTO ', 47: 'TENOR ',
                       48: 'BARITONE ', 49: 'BASS ', 50: 'R&B '}
        labels = [value for key, value in rwc_classes.iteritems() if key in y]

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    # le2 = preprocessing.LabelEncoder()
    # le2.fit(full_classes_names)

    # ANOVA SVM-C
    # 1) anova filter, take N best ranked features
    # 2) svm

    estimators = [("scale", preprocessing.StandardScaler()),
                  ('anova_filter', SelectKBest(chi2, k=100)),
                  ('svm', svm.SVC(decision_function_shape='ovo'))]
    clf = Pipeline(estimators)
    return clf, X, y, labels


def grid_search(clf, X, y):
    params = dict(anova_filter__k=[50, 100],
                  svm__kernel=['rbf'], svm__C=[0.1],
                  svm__degree=[1, 3], svm__gamma=[0.01])
    gs = GridSearchCV(clf, param_grid=params, cv=10, verbose=2)
    gs.fit(X, y)
    print "Best estimator:"
    print gs.best_estimator_
    print "Best parameters:"
    print gs.best_params_
    print "Best score:"
    print gs.best_score_

    y_pred = gs.predict(X)
    y_test = y
    save_results(y_test, y_pred)


def save_results(y_test, y_pred, fold_number=0):
    pickle.dump(y_test, open("y_test_fold{number}.plk".format(number=fold_number), "w"))
    pickle.dump(y_pred, open("y_pred_fold{number}.plk".format(number=fold_number), "w"))
    print classification_report(y_test, y_pred)
    print confusion_matrix(y_test, y_pred)
    print "Micro stats:"
    print precision_recall_fscore_support(y_test, y_pred, average='micro')
    print "Macro stats:"
    print precision_recall_fscore_support(y_test, y_pred, average='macro')
    try:
        visualization.plot_confusion_matrix(confusion_matrix(y_test, y_pred),
                                            title="Test CM fold{number}".format(number=fold_number),
                                            labels=labels)
    except:
        pass


def train_test(clf, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    save_results(y_test, y_pred)


def train_evaluate_stratified(clf, X, y):
    skf = StratifiedKFold(y, n_folds=10)
    for fold_number, (train_index, test_index) in enumerate(skf):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        save_results(y_test, y_pred, fold_number)


if __name__ == "__main__":
    datafile = sys.argv[1]
    dataset = sys.argv[2]

    clf, X, y, labels = feature_preprocessing(datafile, dataset)
    grid_search(clf, X, y)
