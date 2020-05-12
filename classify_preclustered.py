import argparse
import numpy as np
from utils import load_twfeatures, load_pf_hists
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from plot_cm import plot_confusion_matrix
import warnings

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
# from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import OneHotEncoder

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
    subjects_list = [1004]#[1032,1049]#[1004,1006,1007,1019,1020,1023,1032,1034,1038,1046,1048,1049] #1051,1044,1039,1043

if args.study == "REAL":
    path="/media/marcelomdu/Data/GIT_Repos/BEAT-PD/Datasets/REAL/Train/training_data/smartwatch_accelerometer/"
    subjects_list = ['hbv012', 'hbv013', 'hbv022', 'hbv023', 'hbv038','hbv054']#'hbv017', 'hbv051',  'hbv077', 'hbv043', 'hbv014', 'hbv018', 

if not (args.subject == None):
    subjects_list = [args.subject]

#%%
if __name__ == '__main__':

    for subject in subjects_list:
    
        twfeatures = load_twfeatures(path=path,subject=subject)
        sc = StandardScaler()
        pca = PCA(n_components=twfeatures.shape[1],whiten=True)
        clf = KMeans(n_clusters=8)
        twfeatures = sc.fit_transform(twfeatures)
        twfeatures = pca.fit_transform(twfeatures)
        clf = clf.fit(twfeatures)
        
        del twfeatures
        
        pf_hists,labels = load_pf_hists(path=path,subject=subject,classifier=clf,scaler=sc,pca=pca)
        
        
        
        # Select label type
        labels = labels[:,2]

        # Pair wise comparison
        # l1 = 2
        # l2 = 3
        # selected_samples = np.add(enc_labels[:,l1],enc_labels[:,l2])
        # pf_hists = pf_hists[np.where(selected_samples==1)]
        # labels = labels[np.where(selected_samples==1)]
        # Mid cut comparison
        # cut = 2
        # g1 = np.sum(enc_labels[:,:cut],axis=1)
        # labels = g1
        

        
        # labels = labels[:,0]
        
        # pf_hists = StandardScaler().fit_transform(pf_hists)        
        # skb = SelectKBest(k=60)
        # features = skb.fit_transform(pf_hists,labels)
        
        datasets = [[pf_hists,labels]]
        
        # h = .02  # step size in the mesh
        
        # names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
        #           "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        #           "Naive Bayes", "QDA"]      
        
        # classifiers = [
        #     KNeighborsClassifier(3),
        #     SVC(kernel="linear", C=0.025, class_weight='balanced'),
        #     SVC(gamma=2, C=1, class_weight='balanced'),
        #     GaussianProcessClassifier(1.0 * RBF(1.0)),
        #     DecisionTreeClassifier(max_depth=10),
        #     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        #     MLPClassifier(hidden_layer_sizes=(80, ),max_iter=1000,learning_rate='adaptive'),
        #     AdaBoostClassifier(),
        #     GaussianNB(),
        #     QuadraticDiscriminantAnalysis()]
        
        
        k_fold_fits = 10
        # figure = plt.figure(figsize=(27, 9))
        i = 1
        # iterate over datasets
        for ds_cnt, ds in enumerate(datasets):
            # clf = AdaBoostClassifier()
            clf = MLPClassifier(hidden_layer_sizes=(80, ),max_iter=1000,learning_rate='adaptive')
            # preprocess dataset, split into training and test part
            X, y = ds
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=.25, random_state=42)
            # One hot encode labels
            train_labels = np.unique(y_train)
            enc=OneHotEncoder(sparse=False)
            y_train_enc = enc.fit_transform(y_train.reshape(-1,1))
            # y_test = enc.fit_transform(y_test.reshape(-1,1))
            test_pred = {i:list() for i in np.unique(train_labels).astype(int)}
            test_preds = list()
            test_labels = list()
            with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for i in np.unique(train_labels).astype(int):
                        for j in np.unique(train_labels).astype(int):
                            if i!=j:
                                # Pair wise comparison
                                selected_samples = np.add(y_train_enc[:,i],y_train_enc[:,j])
                                features = X_train[np.where(selected_samples==1)]
                                y_true = y_train[np.where(selected_samples==1)]
                                labels = y_train_enc[np.where(selected_samples==1)][:,i]    
                                skb = SelectKBest(k=100)                        
                                skb.fit(features,labels)
                                features = skb.transform(features)
                                test_features = skb.transform(X_test)
                                for _ in range(0,k_fold_fits):
                                    clf.fit(features, labels)                        
                                    test_pred[i].append(clf.predict(test_features))
                                    test_labels.append([i,j])
                        test_preds.append(np.sum(np.stack(test_pred[i]),axis=0))
            
            test_preds = np.stack(test_preds)
            test_preds = np.argmax(test_preds,axis=0)
            
            n_correct = 0
            for i in range(0,test_preds.shape[0]):
                if test_preds[i]==y_test[i]:
                    n_correct += 1
            accuracy = n_correct/test_preds.shape[0]*100
                        
            print(accuracy)
            
    
            class_names = np.array([0,1,2,3])
    
            # Plot non-normalized confusion matrix
            plot_confusion_matrix(y_test, test_preds, classes=class_names,
                                title='Confusion matrix, without normalization')
    
            # Plot normalized confusion matrix
            plot_confusion_matrix(y_test, test_preds, classes=class_names, normalize=True,
                                title='Normalized confusion matrix')
    
            plt.show()
                
            # for name, clf in zip(names, classifiers):
            #     # ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            #     clf.fit(X_train, y_train)
            #     score = clf.score(X_test, y_test)
            #     print("{}\t\t{}".format(name,score))
            #     class_names = [0,1]
            #     np.set_printoptions(precision=2)
            #     # Plot non-normalized confusion matrix
            #     disp = plot_confusion_matrix(clf, X_test, y_test,
            #                                  display_labels=class_names,
            #                                  cmap=plt.cm.Blues,
            #                                  normalize='true')
            #     disp.ax_.set_title(name)     
            #     # print(disp.confusion_matrix)
            #     plt.show()
