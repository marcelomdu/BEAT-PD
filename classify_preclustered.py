import argparse
import numpy as np
from utils import load_twfeatures, load_pf_hists
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_selection import SelectKBest

parser = argparse.ArgumentParser()
parser.add_argument('--study', type=str, default="CIS",
                    help='Study name')
parser.add_argument('--condition', type=str, default="tre",
                    help='Condition for training (med: medication, dys: dyskinesia, tre: tremor)')
parser.add_argument('--subject', default=None,
                    help='Choose single subject for train')

args = parser.parse_args()

if args.study == "CIS":
    path="/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/CIS/Train/training_data/"
    subjects_list = [1032,1049]#[1004,1006,1007,1019,1020,1023,1032,1034,1038,1046,1048,1049] #1051,1044,1039,1043

if args.study == "REAL":
    path="/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/REAL/Train/training_data/smartwatch_accelerometer/"
    subjects_list = ['hbv012', 'hbv013', 'hbv022', 'hbv023', 'hbv038','hbv054']#'hbv017', 'hbv051',  'hbv077', 'hbv043', 'hbv014', 'hbv018', 

if not (args.subject == None):
    subjects_list = [args.subject]

for subject in subjects_list:

    twfeatures = load_twfeatures(path=path,subject=subject)
    sc = StandardScaler()
    # pca = PCA(n_components=twfeatures.shape[1],whiten=True)
    clf = KMeans(n_clusters=10)
    twfeatures = sc.fit_transform(twfeatures)
    # twfeatures = pca.fit_transform(twfeatures)
    clf = clf.fit(twfeatures)
    
    del twfeatures
    
    pf_hists,labels = load_pf_hists(path=path,subject=subject,classifier=clf,scaler=sc)
    
    datasets = [[pf_hists,labels[:,2]]]
    
    h = .02  # step size in the mesh
    
    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]
    
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, class_weight='balanced'),
        SVC(gamma=2, C=1, class_weight='balanced'),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB()]#,
        #QuadraticDiscriminantAnalysis()]
           
    # figure = plt.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=.4, random_state=42)
        for name, clf in zip(names, classifiers):
            # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            print("{}\t\t{}".format(name,score))
            class_names = [0,1]
            np.set_printoptions(precision=2)
            # Plot non-normalized confusion matrix
            disp = plot_confusion_matrix(clf, X_test, y_test,
                                         display_labels=class_names,
                                         cmap=plt.cm.Blues)
            disp.ax_.set_title(name)           
            plt.show()
