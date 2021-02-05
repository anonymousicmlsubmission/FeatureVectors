import numpy as np
from sklearn.tree import _tree
import itertools
import re
from tqdm import tqdm

def accuracy_drop(order, estimator, X_train, y_train, X_test, y_test):
    estimator.fit(X_train, y_train)
    orig_acc = estimator.score(X_test, y_test)
    results = [orig_acc]
    for i in tqdm(range(len(order))):
        ids = order[:i+1]
        estimator.fit(remove_feats(X_train, ids), y_train)
        results.append(estimator.score(remove_feats(X_test, ids), y_test))
    return results

def remove_feats(X, ids):
    if isinstance(ids, int):
        ids = [ids]
    return np.delete(X, ids, axis=-1)

def mean_impute_feats(X, ids, baseline=None):
    if isinstance(ids, int):
        ids = [ids]
    if baseline is None:
        baseline = np.zeros(X.shape[-1])
    X_copy = X.copy()
    X_copy[:, ids] = baseline[ids]
    return X_copy

def return_baseline(X_num, X_cat):
    def return_mode(X_cat):
        modes = []
        for i in range(X_cat.shape[-1]):
            cats, counts = np.unique(X_cat[:,i], axis=0, return_counts=True)
            modes.append(cats[np.argmax(counts)])
        return modes
    return np.concatenate([np.mean(X_num, 0), return_mode(X_cat)])

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)
    
def balanced_dataset(X, y, max_size=None):
    counts = np.bincount(y)
    minority = np.min(counts)
    idxs = []
    for label in np.unique(y):
        idxs.extend(np.random.choice(np.where(y == label)[0], minority, replace=False))
    idxs = np.random.permutation(idxs)
    if max_size:
        idxs = idxs[:max_size]
    return X[idxs], y[idxs]

def powerset(xs, min_size=1, max_size=None):
    
    subsets = []
    if max_size is None:
        max_size = len(xs)
    for i in range(min_size, max_size + 1):
        for s in itertools.combinations(xs, i):
            subsets.append(np.sort(s))
    return subsets

def find_features(term, features_dic):
    
    results = []
    for key in features_dic:
        if term in features_dic[key]:
            results.append((key, features_dic[key]))
    return results

def cooccurance_matrix(rules, num_features, window_size=1):
    
    cm = np.zeros((num_features, num_features))
    for rule in rules:
        group = re.findall('X\d\d\d\d\d', rule)
        if len(group) < 2 * window_size + 1:
            continue
        for i in range(len(group)):
            context = group[max(i - window_size, 0): i] + group[i+1: (i+1+window_size)]
            for f in context:
                cm[int(group[i][1:]), int(f[1:])] += 1
    return cm


    
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def normalize_angles(vectors):
    

    
    rho, phi = cart2pol(vectors[:, 0], vectors[:, 1])
    phi -= np.mean(phi)
    x, y = pol2cart(rho, phi)
    return np.stack([x, y], -1)