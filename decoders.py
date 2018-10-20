import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from scipy.stats import sem
import scipy.io
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from numpy.random import permutation
from numpy.random import choice
from sklearn.linear_model import LinearRegression
#~ import imbalanced_data
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
import scipy.stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import preprocessing


class linear_svm:
    def __init__(self,feat,clase):
        self.feat=preprocessing.scale(feat)
        self.clase=clase
        self.clase_unique=np.unique(self.clase)
        self.prior=len(self.clase[self.clase==1])/float(len(self.clase))
        self.cv=5
        
    def svm(self):
        perf=np.zeros((self.cv))
        perf_train=np.zeros((self.cv))
        wei=np.zeros((self.cv,len(self.feat[0])+1,1))
        skf=StratifiedKFold(self.clase,self.cv)
        g=0
        for train,test in skf: 
            X_train=self.feat[train]
            X_test=self.feat[test]
            y_train=self.clase[train]
            y_test=self.clase[test]
            supp=LinearSVC(dual=True)
            trainning=supp.fit(X_train,y_train)
            perf[g]=supp.score(X_test,y_test)
            perf_train[g]=supp.score(X_train,y_train)
            wei[g,:,0]=np.append(supp.coef_[0],supp.intercept_)
            g=g+1
        performance=np.mean(perf)
        performance_train=np.mean(perf_train)
        weights=np.mean(wei,axis=0)
        output={'performance':performance,'performance_train':performance_train,'weights':weights}
        return output

    def svm_shuffled(self,n):
        perf=np.zeros((self.cv,n))
        perf_train=np.zeros((self.cv,n))
        wei=np.zeros((self.cv,n,len(self.feat[0])+1,1))
        for k in range(n):
            clase_s=permutation(self.clase)
            skf=StratifiedKFold(clase_s,self.cv) 
            g=0
            for train,test in skf: 
                X_train=self.feat[train]
                X_test=self.feat[test]
                y_train=clase_s[train]
                y_test=clase_s[test]
                supp=LinearSVC(dual=True)
                trainning=supp.fit(X_train,y_train)
                perf[g,k]=supp.score(X_test,y_test)
                perf_train[g,k]=supp.score(X_train,y_train)
                wei[g,k,:,0]=np.append(supp.coef_[0],supp.intercept_)
                g=g+1
        performance_distr=np.mean(perf,axis=0)
        performance_train_distr=np.mean(perf_train,axis=0)
        weights_distr=np.mean(wei,axis=0)
        output={'performance_distr':performance_distr,'performance_train_distr':performance_train_distr,'weights_distr':weights_distr}
        return output

    def svm_uniform_prior(self,n_unif,method): 
        perf=np.zeros((n_unif,self.cv))
        perf_th=np.zeros((n_unif,self.cv))
        perf_train=np.zeros((n_unif,self.cv))
        wei=np.zeros((n_unif,self.cv,len(self.feat[0])+1,1))
        if method=='oversampling':
            p_in=imbalanced_data.oversampling(self.clase,n_unif)
            index_balanced=p_in.index
        if method=='undersampling':
            p_in=imbalanced_data.undersampling(self.clase,n_unif)
            index_balanced=p_in.index
        for i in range(len(index_balanced)):
            clase_balanced=self.clase[index_balanced[i]]
            feat_balanced=self.feat[index_balanced[i]]            
            skf=StratifiedKFold(clase_balanced,self.cv)  
            g=0
            for train,test in skf: 
                X_train=feat_balanced[train]
                X_test=feat_balanced[test]
                y_train=clase_balanced[train]
                y_test=clase_balanced[test]
                supp=LinearSVC(dual=True)
                trainning=supp.fit(X_train,y_train)
                perf[i,g]=supp.score(X_test,y_test)
                perf_train[i,g]=supp.score(X_train,y_train)
                wei[i,g,:,0]=np.append(supp.coef_[0],supp.intercept_)
                g=g+1
        performance=np.mean(perf)
        performance_train=np.mean(perf_train)
        weights=np.mean(wei,axis=(0,1))
        output={'performance':performance,'performance_train':performance_train,'weights':weights}
        return output

    def svm_uniform_prior_shuffled(self,n_unif,n,method): 
        perf=np.zeros((n_unif,self.cv,n))
        wei=np.zeros((n_unif,self.cv,n,len(self.feat[0])+1,1))
        for k in range(n):
            clase_s=permutation(self.clase)
            if method=='oversampling':
                p_in=imbalanced_data.oversampling(clase_s,n_unif)
                index_balanced=p_in.index
            if method=='undersampling':
                p_in=imbalanced_data.undersampling(clase_s,n_unif)
                index_balanced=p_in.index
            for i in range(len(index_balanced)):
                clase_balanced=clase_s[index_balanced[i]]
                feat_balanced=self.feat[index_balanced[i]]
                skf=StratifiedKFold(clase_balanced,self.cv)  
                g=0
                for train,test in skf: 
                    X_train=feat_balanced[train]
                    X_test=feat_balanced[test]
                    y_train=clase_balanced[train]
                    y_test=clase_balanced[test]
                    supp=LinearSVC(dual=True)
                    trainning=supp.fit(X_train,y_train)
                    perf[i,g,k]=supp.score(X_test,y_test)
                    wei[i,g,k,:,0]=np.append(supp.coef_[0],supp.intercept_)
                    g=g+1
        performance_distr=np.mean(perf,axis=(0,1))
        weights_distr=np.mean(wei,axis=(0,1))
        output={'performance_distr':performance_distr,'weights_distr':weights_distr}
        return output

class svm:
    def __init__(self,feat,clase,regularization,kernel):
        self.feat=preprocessing.scale(feat)
        self.clase=clase
        self.clase_unique=np.unique(self.clase)
        self.prior=len(self.clase[self.clase==1])/float(len(self.clase))
        self.cv=5
        self.regularization=regularization
        self.kernel=kernel

    def svm(self):
        perf=np.zeros((self.cv))
        perf_train=np.zeros((self.cv))
        skf=StratifiedKFold(self.clase,self.cv)
        g=0
        for train,test in skf: 
            X_train=self.feat[train]
            X_test=self.feat[test]
            y_train=self.clase[train]
            y_test=self.clase[test]
            supp=SVC(C=self.regularization,kernel=self.kernel)
            trainning=supp.fit(X_train,y_train)
            perf[g]=supp.score(X_test,y_test)
            perf_train[g]=supp.score(X_train,y_train)
            g=g+1
        performance=np.mean(perf)
        performance_train=np.mean(perf_train)
        output={'performance':performance,'performance_train':performance_train}
        return output

    def svm_shuffled(self,n):
        perf=np.zeros((self.cv,n))
        perf_train=np.zeros((self.cv,n))
        for k in range(n):
            clase_s=permutation(self.clase)
            skf=StratifiedKFold(clase_s,self.cv) 
            g=0
            for train,test in skf: 
                X_train=self.feat[train]
                X_test=self.feat[test]
                y_train=clase_s[train]
                y_test=clase_s[test]
                supp=SVC(C=self.regularization,kernel=self.kernel)
                trainning=supp.fit(X_train,y_train)
                perf[g,k]=supp.score(X_test,y_test)
                perf_train[g,k]=supp.score(X_train,y_train)
                g=g+1
        performance_distr=np.mean(perf,axis=0)
        performance_train_distr=np.mean(perf_train,axis=0)
        output={'performance_distr':performance_distr,'performance_train_distr':performance_train_distr}
        return output

    def svm_uniform_prior(self,n_unif,method): 
        perf=np.zeros((n_unif,self.cv))
        perf_train=np.zeros((n_unif,self.cv))
        if method=='oversampling':
            p_in=imbalanced_data.oversampling(self.clase,n_unif)
            index_balanced=p_in.index
        if method=='undersampling':
            p_in=imbalanced_data.undersampling(self.clase,n_unif)
            index_balanced=p_in.index
        for i in range(len(index_balanced)):
            clase_balanced=self.clase[index_balanced[i]]
            feat_balanced=self.feat[index_balanced[i]]            
            skf=StratifiedKFold(clase_balanced,self.cv)  
            g=0
            for train,test in skf: 
                X_train=feat_balanced[train]
                X_test=feat_balanced[test]
                y_train=clase_balanced[train]
                y_test=clase_balanced[test]
                supp=SVC(C=self.regularization,kernel=self.kernel)
                trainning=supp.fit(X_train,y_train)
                perf[i,g]=supp.score(X_test,y_test)
                perf_train[i,g]=supp.score(X_train,y_train)
                g=g+1
        performance=np.mean(perf)
        performance_train=np.mean(perf_train)
        output={'performance':performance,'performance_train':performance_train}
        return output

    def svm_uniform_prior_shuffled(self,n_unif,n,method): 
        perf=np.zeros((n_unif,self.cv,n))
        for k in range(n):
            clase_s=permutation(self.clase)
            if method=='oversampling':
                p_in=imbalanced_data.oversampling(clase_s,n_unif)
                index_balanced=p_in.index
            if method=='undersampling':
                p_in=imbalanced_data.undersampling(clase_s,n_unif)
                index_balanced=p_in.index
            for i in range(len(index_balanced)):
                clase_balanced=clase_s[index_balanced[i]]
                feat_balanced=self.feat[index_balanced[i]]
                skf=StratifiedKFold(clase_balanced,self.cv)  
                g=0
                for train,test in skf: 
                    X_train=feat_balanced[train]
                    X_test=feat_balanced[test]
                    y_train=clase_balanced[train]
                    y_test=clase_balanced[test]
                    supp=SVC(C=self.regularization,kernel=self.kernel)
                    trainning=supp.fit(X_train,y_train)
                    perf[i,g,k]=supp.score(X_test,y_test)
                    g=g+1
        performance_distr=np.mean(perf,axis=(0,1))
        output={'performance_distr':performance_distr}
        return output

class logregress:
    def __init__(self,feat,clase, regularization=10**5, cv=5, cv_shuffle=True,
        balance_classes=True, random_state=0):
        """Initalize logistic regression object
        
        cv : number of folds
        cv_shuffle : shuffle the trials included in each fold
            Used to set 'shuffle' in StratifiedKFold
        balance_classes : whether to set balance classes by setting
            'class_weight' to 'balanced' in LogisticRegression
        random_state : sent to StratifiedKFold for shuffling
        
        """
        self.feat=feat
        self.clase=clase
        self.clase_unique=np.unique(self.clase)
        self.prior=len(self.clase[self.clase==1])/float(len(self.clase))
        self.cv = cv
        self.cv_shuffle = cv_shuffle
        self.regularization = regularization
        self.balance_classes = balance_classes
        self.random_state = random_state
        if regularization == 0.0:
            print 'la regularization no deberia ser 0'
            self.regularization=10**5

    def logregress(self):
        """Run cross-validated logistic regression 
        
        Uses self.feat and self.clase as the features and labels. Uses
        StratifiedKFold to generated the test and train sets. For each
        fold, fits on the train set and tests on the test set.
        
        Returns: dict
            'performance' : performance on test set, meaned over folds
            'performance_train': same, but for training set
            'weights': coefficients, meaned over folds
                The last entry will be the intercept
            'test_idxs': list of length n_folds (default 4)
                Each entry is an array of indices into features and labels
                that were tested on this fold.
            'decision_function': decision function for all tested indices
                This is in the same order as test_idxs, but concatenated
                over folds
            'predict_probability': probability
                This is in the same order as test_idxs, but concatenated
                over folds            
        """
        perf=np.zeros((self.cv))
        perf_train=np.zeros((self.cv))
        wei=np.zeros((self.cv,len(self.feat[0])+1,1))
        dec_function=np.array([])
        predict_proba=np.array([])
        skf=StratifiedKFold(n_splits=self.cv, shuffle=self.cv_shuffle,
            random_state=self.random_state)
        test_idxs = []
        
        # Iterate over folds
        g=0
        for train,test in skf.split(self.feat, self.clase): 
            # Split out test and train sets
            X_train=self.feat[train]
            X_test=self.feat[test]
            y_train=self.clase[train]
            y_test=self.clase[test]
            
            # Initalize fitter
            class_weight = 'balanced' if self.balance_classes else None
            log=LogisticRegression(
                C=(1.0/self.regularization),
                class_weight=class_weight,
            )
            
            # Fit
            trainning=log.fit(X_train,y_train)
            
            # Iteratively stack the decision function and predict proba
            dec_function=np.hstack((dec_function,log.decision_function(X_test)))
            predict_proba=np.hstack((predict_proba,log.predict_proba(X_test)[:,1]))
            
            # Store the performance on test and train
            perf[g]=log.score(X_test,y_test)
            perf_train[g]=log.score(X_train,y_train)
            
            # Store the train indices (which match up with dec_function and 
            # predict_proba)
            test_idxs.append(test)
            
            # Store the weights
            wei[g,:,0]=np.append(log.coef_[0],log.intercept_)
            
            # Increment fold count
            g=g+1
        
        # Mean performance and weights over folds
        performance=np.mean(perf)
        performance_train=np.mean(perf_train)
        weights=np.mean(wei,axis=0)
        
        output = {
            'performance': performance,
            'performance_train': performance_train,
            'weights': weights,
            'decision_function': dec_function,
            'predict_probability': predict_proba,
            'test_idxs': test_idxs,
        }
        return output

    def logregress_shuffled(self,n):
        perf=np.zeros((self.cv,n))
        perf_train=np.zeros((self.cv,n))
        wei=np.zeros((self.cv,n,len(self.feat[0])+1,1))
        for k in range(n):
            clase_s=permutation(self.clase)
            skf=StratifiedKFold(clase_s,self.cv) 
            g=0
            for train,test in skf: 
                X_train=self.feat[train]
                X_test=self.feat[test]
                y_train=clase_s[train]
                y_test=clase_s[test]
                log=LogisticRegression(C=(1.0/self.regularization))
                trainning=log.fit(X_train,y_train)
                perf[g,k]=log.score(X_test,y_test)
                perf_train[g,k]=log.score(X_train,y_train)
                wei[g,k,:,0]=np.append(log.coef_[0],log.intercept_)
                g=g+1
        performance_distr=np.mean(perf,axis=0)
        performance_train_distr=np.mean(perf_train,axis=0)
        weights_distr=np.mean(wei,axis=0)
        output={'performance_distr':performance_distr,'performance_train_distr':performance_train_distr,'weights_distr':weights_distr}
        return output

    def logregress_uniform_prior(self,n_unif,method): 
        perf=np.zeros((n_unif,self.cv))
        perf_th=np.zeros((n_unif,self.cv))
        perf_train=np.zeros((n_unif,self.cv))
        wei=np.zeros((n_unif,self.cv,len(self.feat[0])+1,1))
        if method=='oversampling':
            p_in=imbalanced_data.oversampling(self.clase,n_unif)
            index_balanced=p_in.index
        if method=='undersampling':
            p_in=imbalanced_data.undersampling(self.clase,n_unif)
            index_balanced=p_in.index
        for i in range(len(index_balanced)):
            clase_balanced=self.clase[index_balanced[i]]
            feat_balanced=self.feat[index_balanced[i]]            
            skf=StratifiedKFold(clase_balanced,self.cv)  
            g=0
            for train,test in skf: 
                X_train=feat_balanced[train]
                X_test=feat_balanced[test]
                y_train=clase_balanced[train]
                y_test=clase_balanced[test]
                log=LogisticRegression(C=(1.0/self.regularization))
                trainning=log.fit(X_train,y_train)
                perf[i,g]=log.score(X_test,y_test)
                perf_train[i,g]=log.score(X_train,y_train)
                wei[i,g,:,0]=np.append(log.coef_[0],log.intercept_)
                g=g+1
        performance=np.mean(perf)
        performance_th=np.mean(perf_th)
        performance_train=np.mean(perf_train)
        weights=np.mean(wei,axis=(0,1))
        output={'performance':performance,'performance_train':performance_train,'weights':weights}
        return output

    def logregress_uniform_prior_shuffled(self,n_unif,n,method): 
        perf=np.zeros((n_unif,self.cv,n))
        wei=np.zeros((n_unif,self.cv,n,len(self.feat[0])+1,1))
        for k in range(n):
            clase_s=permutation(self.clase)
            if method=='oversampling':
                p_in=imbalanced_data.oversampling(clase_s,n_unif)
                index_balanced=p_in.index
            if method=='undersampling':
                p_in=imbalanced_data.undersampling(clase_s,n_unif)
                index_balanced=p_in.index
            for i in range(len(index_balanced)):
                clase_balanced=clase_s[index_balanced[i]]
                feat_balanced=self.feat[index_balanced[i]]
                skf=StratifiedKFold(clase_balanced,self.cv)  
                g=0
                for train,test in skf: 
                    X_train=feat_balanced[train]
                    X_test=feat_balanced[test]
                    y_train=clase_balanced[train]
                    y_test=clase_balanced[test]
                    log=LogisticRegression(C=(1.0/self.regularization))
                    trainning=log.fit(X_train,y_train)
                    perf[i,g,k]=log.score(X_test,y_test)
                    wei[i,g,k,:,0]=np.append(log.coef_[0],log.intercept_)
                    g=g+1
        performance_distr=np.mean(perf,axis=(0,1))
        weights_distr=np.mean(wei,axis=(0,1))
        output={'performance_distr':performance_distr,'weights_distr':weights_distr}
        return output

class logregress_multinomial:
    def __init__(self,feat,clase,regularization):
        self.clase=clase
        self.feat=feat
        self.outputs_unique=np.unique(self.clase)
        self.num_outputs=len(self.outputs_unique)
        self.chance=(1.0/self.num_outputs)
        self.prior=max(np.array([np.sum(self.clase==i) for i in self.outputs_unique]))/float(len(self.clase))
        self.cv=5
        self.regularization=regularization

    def logregress(self):
        perf=np.zeros((self.cv))
        perf_train=np.zeros((self.cv))
        wei=np.zeros((self.cv,self.num_outputs,len(self.feat[0])+1))
        skf=StratifiedKFold(self.clase,self.cv)
        g=0
        for train,test in skf: 
            X_train=self.feat[train]
            X_test=self.feat[test]
            y_train=self.clase[train]
            y_test=self.clase[test]
            log=LogisticRegression(C=(1.0/self.regularization),multi_class='multinomial',solver='lbfgs')
            trainning=log.fit(X_train,y_train)
            perf[g]=log.score(X_test,y_test)
            perf_train[g]=log.score(X_train,y_train)
            offset=np.reshape(log.intercept_,(1,len(log.intercept_)))
            #wei[g]=np.concatenate((log.coef_,offset),axis=1) # hay que incorporar los weights del intercept
            g=g+1
        performance=np.mean(perf)
        performance_train=np.mean(perf_train)
        weights=np.mean(wei,axis=0)
        output={'performance':performance,'performance_train':performance_train,'weights':weights}
        return output

    def logregress_shuffled(self,n):
        perf=np.zeros((self.cv,n))
        perf_train=np.zeros((self.cv,n))
        wei=np.zeros((self.cv,n,self.num_outputs,len(self.feat[0])+1))
        for k in range(n):  
            clase_s=permutation(self.clase)
            skf=StratifiedKFold(len(clase_s),self.cv) 
            g=0
            for train,test in skf: 
                X_train=self.feat[train]
                X_test=self.feat[test]
                y_train=self.clase[train]
                y_test=self.clase[test]
                log=LogisticRegression(C=(1.0/self.regularization),multi_class='multinomial',solver='lbfgs')
                trainning=log.fit(X_train,y_train)
                perf[g,k]=log.score(X_test,y_test)
                perf_train[g,k]=log.score(X_train,y_train)
                offset=np.reshape(log.intercept_,(1,len(log.intercept_)))
                wei[g,k]=np.concatenate((log.coef_,offset),axis=1)
                g=g+1
        performance_distr=np.mean(perf,axis=0)
        performance_train_distr=np.mean(perf_train,axis=0)
        weights_distr=np.mean(wei,axis=0)
        output={'performance_distr':performance_distr,'performance_train_distr':performance_train_distr,'weights_distr':weights_distr}
        return output

    def logregress_uniform_prior(self,n_unif,method):
        perf=np.zeros((n_unif,self.cv))
        perf_categories=np.zeros((n_unif,self.cv,self.num_outputs))
        perf_train=np.zeros((n_unif,self.cv))
        wei=np.zeros((n_unif,self.cv,self.num_outputs,len(self.feat[0])+1))
        if method=='undersampling':
            p_in=imbalanced_data.undersampling_multinomial(self.clase,n_unif)
            index_balanced=p_in.index
        for i in range(len(index_balanced)):
            clase_balanced=self.clase[index_balanced[i]]
            feat_balanced=self.feat[index_balanced[i]]
            skf=StratifiedKFold(clase_balanced,self.cv)  
            g=0
            for train,test in skf: 
                X_train=feat_balanced[train]
                X_test=feat_balanced[test]
                y_train=clase_balanced[train]
                y_test=clase_balanced[test]
                log=LogisticRegression(C=(1.0/self.regularization),multi_class='multinomial',solver='lbfgs')
                trainning=log.fit(X_train,y_train)
                perf[i,g]=log.score(X_test,y_test)
                perf_train[i,g]=log.score(X_train,y_train)
                offset=np.reshape(log.intercept_,(len(log.intercept_),1))
                wei[i,g]=np.concatenate((log.coef_,offset),axis=1)
                for hh in range(self.num_outputs):
                    y_test_cir=y_test[y_test==self.outputs_unique[hh]]
                    X_test_cir=X_test[y_test==self.outputs_unique[hh]]
                    pred_cir=log.predict(X_test_cir)
                    perf_cir=np.sum(pred_cir==y_test_cir)/float(len(y_test_cir))
                    perf_categories[i,g,hh]=perf_cir
                g=g+1
        performance=np.mean(perf)
        performance_categories=np.mean(perf_categories,axis=(0,1))
        performance_train=np.mean(perf_train)
        weights=np.mean(wei,axis=(0,1))
        output={'performance':performance,'performance_categories':performance_categories,'performance_train':performance_train,'weights':weights}
        return output

    def logregress_uniform_prior_shuffled(self,n_unif,n,method): 
        perf=np.zeros((n_unif,self.cv,n))
        wei=np.zeros((n_unif,self.cv,n,self.num_outputs,len(self.feat[0])+1))
        for k in range(n):
            clase_s=permutation(self.clase)
            if method=='undersampling':
                p_in=imbalanced_data.undersampling_multinomial(clase_s,n_unif)
                index_balanced=p_in.index
            for i in range(len(index_balanced)):
                clase_balanced=clase_s[index_balanced[i]]
                feat_balanced=self.feat[index_balanced[i]]
                skf=StratifiedKFold(clase_balanced,self.cv)  
                g=0
                for train,test in skf: 
                    X_train=feat_balanced[train]
                    X_test=feat_balanced[test]
                    y_train=clase_balanced[train]
                    y_test=clase_balanced[test]
                    log=LogisticRegression(C=(1.0/self.regularization),multi_class='multinomial',solver='lbfgs')
                    trainning=log.fit(X_train,y_train)
                    perf[i,g,k]=log.score(X_test,y_test)
                    offset=np.reshape(log.intercept_,(len(log.intercept_),1))
                    wei[i,g,k]=np.concatenate((log.coef_,offset),axis=1)
                    g=g+1
        performance_distr=np.mean(perf,axis=(0,1))
        weights_distr=np.mean(wei,axis=(0,1))
        output={'performance_distr':performance_distr,'weights_distr':weights_distr}
        return output

class lda:
    def __init__(self,feat,clase):
        self.feat=feat
        self.clase=clase
        self.clase_unique=np.unique(self.clase)
        self.cv=5

    def lda(self):
        perf=np.zeros((self.cv))
        perf_train=np.zeros((self.cv))
        wei=np.zeros((self.cv,len(self.feat[0])+1,1))
        skf=StratifiedKFold(self.clase,self.cv)
        g=0
        for train,test in skf: 
            X_train=self.feat[train]
            X_test=self.feat[test]
            y_train=self.clase[train]
            y_test=self.clase[test]
            log=LinearDiscriminantAnalysis()
            trainning=log.fit(X_train,y_train)
            perf[g]=log.score(X_test,y_test)
            perf_train[g]=log.score(X_train,y_train)
            wei[g,:,0]=np.append(log.coef_[0],log.intercept_)
            g=g+1
        performance=np.mean(perf)
        performance_train=np.mean(perf_train)
        weights=np.mean(wei,axis=0)
        output={'performance':performance,'performance_train':performance_train,'weights':weights}
        return output

    def lda_shuffled(self,n):
        perf=np.zeros((self.cv,n))
        perf_train=np.zeros((self.cv,n))
        wei=np.zeros((self.cv,n,len(self.feat[0])+1,1))
        for k in range(n):
            clase_s=permutation(self.clase)
            skf=StratifiedKFold(clase_s,self.cv) 
            g=0
            for train,test in skf: 
                X_train=self.feat[train]
                X_test=self.feat[test]
                y_train=clase_s[train]
                y_test=clase_s[test]
                log=LinearDiscriminantAnalysis()
                trainning=log.fit(X_train,y_train)
                perf[g,k]=log.score(X_test,y_test)
                perf_train[g,k]=log.score(X_train,y_train)
                wei[g,k,:,0]=np.append(log.coef_[0],log.intercept_)
                g=g+1
        performance_distr=np.mean(perf,axis=0)
        performance_train_distr=np.mean(perf_train,axis=0)
        weights_distr=np.mean(wei,axis=0)
        output={'performance_distr':performance_distr,'performance_train_distr':performance_train_distr,'weights_distr':weights_distr}
        return output

    def lda_uniform_prior(self,n_unif,method): 
        perf=np.zeros((n_unif,self.cv))
        perf_th=np.zeros((n_unif,self.cv))
        perf_train=np.zeros((n_unif,self.cv))
        wei=np.zeros((n_unif,self.cv,len(self.feat[0])+1,1))
        if method=='oversampling':
            p_in=imbalanced_data.oversampling(self.clase,n_unif)
            index_balanced=p_in.index
        if method=='undersampling':
            p_in=imbalanced_data.undersampling(self.clase,n_unif)
            index_balanced=p_in.index
        for i in range(len(index_balanced)):
            clase_balanced=self.clase[index_balanced[i]]
            feat_balanced=self.feat[index_balanced[i]]            
            skf=StratifiedKFold(clase_balanced,self.cv)  
            g=0
            for train,test in skf: 
                X_train=feat_balanced[train]
                X_test=feat_balanced[test]
                y_train=clase_balanced[train]
                y_test=clase_balanced[test]
                log=LinearDiscriminantAnalysis()
                trainning=log.fit(X_train,y_train)
                perf[i,g]=log.score(X_test,y_test)
                perf_th[i,g]=1.0/((np.std(log.decision_function(X_test)))**2)
                perf_train[i,g]=log.score(X_train,y_train)
                wei[i,g,:,0]=np.append(log.coef_[0],log.intercept_)
                g=g+1
        performance=np.mean(perf)
        performance_th=np.mean(perf_th)
        performance_train=np.mean(perf_train)
        weights=np.mean(wei,axis=(0,1))
        output={'performance':performance,'performance_th':performance_th,'performance_train':performance_train,'weights':weights}
        return output

    def lda_uniform_prior_shuffled(self,n_unif,n,method): 
        perf=np.zeros((n_unif,self.cv,n))
        wei=np.zeros((n_unif,self.cv,n,len(self.feat[0])+1,1))
        for k in range(n):
            clase_s=permutation(self.clase)
            if method=='oversampling':
                p_in=imbalanced_data.oversampling(clase_s,n_unif)
                index_balanced=p_in.index
            if method=='undersampling':
                p_in=imbalanced_data.undersampling(clase_s,n_unif)
                index_balanced=p_in.index
            for i in range(len(index_balanced)):
                clase_balanced=clase_s[index_balanced[i]]
                feat_balanced=self.feat[index_balanced[i]]
                skf=StratifiedKFold(clase_balanced,self.cv)  
                g=0
                for train,test in skf: 
                    X_train=feat_balanced[train]
                    X_test=feat_balanced[test]
                    y_train=clase_balanced[train]
                    y_test=clase_balanced[test]
                    log=LinearDiscriminantAnalysis()
                    trainning=log.fit(X_train,y_train)
                    perf[i,g,k]=log.score(X_test,y_test)
                    wei[i,g,k,:,0]=np.append(log.coef_[0],log.intercept_)
                    g=g+1
        performance_distr=np.mean(perf,axis=(0,1))
        weights_distr=np.mean(wei,axis=(0,1))
        output={'performance_distr':performance_distr,'weights_distr':weights_distr}
        return output

class qda:
    def __init__(self,feat,clase):
        self.feat=feat
        self.clase=clase
        self.clase_unique=np.unique(self.clase)
        self.cv=5

    def qda(self):
        perf=np.zeros((self.cv))
        perf_train=np.zeros((self.cv))
        skf=StratifiedKFold(self.clase,self.cv)
        g=0
        for train,test in skf: 
            X_train=self.feat[train]
            X_test=self.feat[test]
            y_train=self.clase[train]
            y_test=self.clase[test]
            log=QuadraticDiscriminantAnalysis()
            trainning=log.fit(X_train,y_train)
            perf[g]=log.score(X_test,y_test)
            perf_train[g]=log.score(X_train,y_train)
            g=g+1
        performance=np.mean(perf)
        performance_train=np.mean(perf_train)
        output={'performance':performance,'performance_train':performance_train}
        return output

    def qda_shuffled(self,n):
        perf=np.zeros((self.cv,n))
        perf_train=np.zeros((self.cv,n))
        for k in range(n):
            clase_s=permutation(self.clase)
            skf=StratifiedKFold(clase_s,self.cv) 
            g=0
            for train,test in skf: 
                X_train=self.feat[train]
                X_test=self.feat[test]
                y_train=clase_s[train]
                y_test=clase_s[test]
                log=QuadraticDiscriminantAnalysis()
                trainning=log.fit(X_train,y_train)
                perf[g,k]=log.score(X_test,y_test)
                perf_train[g,k]=log.score(X_train,y_train)
                g=g+1
        performance_distr=np.mean(perf,axis=0)
        performance_train_distr=np.mean(perf_train,axis=0)
        output={'performance_distr':performance_distr,'performance_train_distr':performance_train_distr}
        return output

    def qda_uniform_prior(self,n_unif,method): 
        perf=np.zeros((n_unif,self.cv))
        perf_th=np.zeros((n_unif,self.cv))
        perf_train=np.zeros((n_unif,self.cv))
        if method=='oversampling':
            p_in=imbalanced_data.oversampling(self.clase,n_unif)
            index_balanced=p_in.index
        if method=='undersampling':
            p_in=imbalanced_data.undersampling(self.clase,n_unif)
            index_balanced=p_in.index
        for i in range(len(index_balanced)):
            clase_balanced=self.clase[index_balanced[i]]
            feat_balanced=self.feat[index_balanced[i]]            
            skf=StratifiedKFold(clase_balanced,self.cv)  
            g=0
            for train,test in skf: 
                X_train=feat_balanced[train]
                X_test=feat_balanced[test]
                y_train=clase_balanced[train]
                y_test=clase_balanced[test]
                log=QuadraticDiscriminantAnalysis()
                trainning=log.fit(X_train,y_train)
                perf[i,g]=log.score(X_test,y_test)
                perf_th[i,g]=1.0/((np.std(log.decision_function(X_test)))**2)
                perf_train[i,g]=log.score(X_train,y_train)
                g=g+1
        performance=np.mean(perf)
        performance_th=np.mean(perf_th)
        performance_train=np.mean(perf_train)
        output={'performance':performance,'performance_th':performance_th,'performance_train':performance_train}
        return output

    def qda_uniform_prior_shuffled(self,n_unif,n,method): 
        perf=np.zeros((n_unif,self.cv,n))
        for k in range(n):
            clase_s=permutation(self.clase)
            if method=='oversampling':
                p_in=imbalanced_data.oversampling(clase_s,n_unif)
                index_balanced=p_in.index
            if method=='undersampling':
                p_in=imbalanced_data.undersampling(clase_s,n_unif)
                index_balanced=p_in.index
            for i in range(len(index_balanced)):
                clase_balanced=clase_s[index_balanced[i]]
                feat_balanced=self.feat[index_balanced[i]]
                skf=StratifiedKFold(clase_balanced,self.cv)  
                g=0
                for train,test in skf: 
                    X_train=feat_balanced[train]
                    X_test=feat_balanced[test]
                    y_train=clase_balanced[train]
                    y_test=clase_balanced[test]
                    log=QuadraticDiscriminantAnalysis()
                    trainning=log.fit(X_train,y_train)
                    perf[i,g,k]=log.score(X_test,y_test)
                    g=g+1
        performance_distr=np.mean(perf,axis=(0,1))
        output={'performance_distr':performance_distr}
        return output

class linregress_ols:
    def __init__(self,regressors,target):
        self.regressors=regressors
        self.target=target
        self.fit_intercept=True
        self.normalize=False
        self.test_size=0.2
        self.n_iter=100
        self.n_iter_sh=5

    def linregress(self):
        perform=[]
        perform_train=[]
        weights=[]
        weight0=[]
        skf=ShuffleSplit(len(self.target),n_iter=self.n_iter,test_size=self.test_size)  #  KFold(n=len(self.target),n_folds=5)
        for train,test in skf: 
            X_train=self.regressors[train]
            X_test=self.regressors[test]
            y_train=self.target[train]
            y_test=self.target[test]
            linr=LinearRegression(fit_intercept=self.fit_intercept,normalize=self.normalize)    
            trainning=linr.fit(X_train,y_train)
            prediccion=linr.predict(X_test)
	    perf=linr.score(X_test,y_test)
            prediccion_trian=linr.predict(X_train)
            perf_train=linr.score(X_train,y_train)
            perform.append(perf) 
            perform_train.append(perf_train)
            weights.append(linr.coef_)
            weight0.append(linr.intercept_)
        performance=np.mean(perform)
        performance_train=np.mean(perform_train)
        weights=np.mean(weights,axis=0)
        weight0=np.mean(weight0)
        output={'performance':performance,'performance_train':performance_train,'weights':weights,'weight0':weight0}
        return output

    def linregress_shuffled(self,n):
        perform=[]
        weights=[]
        weight0=[] 
        for k in range(n):  
            perfor=[]
            wei=[]
            wei0=[]
            target_s=permutation(self.target)
            skf=ShuffleSplit(len(target_s),n_iter=self.n_iter_sh,test_size=self.test_size)       
            for train,test in skf: 
                X_train=self.regressors[train]
                X_test=self.regressors[test]
                y_train=target_s[train]
                y_test=target_s[test]
                linr=LinearRegression(fit_intercept=self.fit_intercept,normalize=self.normalize)
                trainning=linr.fit(X_train,y_train)
                prediccion=linr.predict(X_test)
                perf=linr.score(X_test,y_test)
                perfor.append(perf) 
                wei.append(linr.coef_)
                wei0.append(linr.intercept_)
            perform.append(np.mean(perfor))
            weights.append(np.mean(wei,axis=0))
            weight0.append(np.mean(wei0))
        performance_distr=np.array(perform)
        weights_distr=np.array(weights)
        weights=np.mean(weights,axis=0)
        weight0=np.mean(weight0)
        output={'performance_distr':performance_distr,'weights':weights,'weight0':weight0,'weights_distr':weights_distr}
        return output

class logregress_statsmodels:
    def __init__(self,feat,clase,regularization):
        self.feat=sm.add_constant(feat,prepend=False) # En los weights el ultimo valor es el del intercept
        self.clase=clase
        self.clase[self.clase==-1]=0
        self.cv=5
        self.regularization=regularization

    def logregress(self):
        perf=np.zeros((self.cv))
        perf_train=np.zeros((self.cv))
        wei=np.zeros((self.cv,len(self.feat[0])))
        skf=StratifiedKFold(self.clase,self.cv)
        g=0
        for train,test in skf: 
            X_train=self.feat[train]
            X_test=self.feat[test]
            y_train=self.clase[train]
            y_test=self.clase[test]
            lr=sm.Logit(endog=y_train,exog=X_train)
            model_pre=lr.fit(disp=False)
            model=lr.fit_regularized(start_params=model_pre.params,alpha=self.regularization,disp=False)
            predict=(model.predict(exog=X_test))>0.5
            predict_train=(model.predict(exog=X_train))>0.5
            perf[g]=np.sum(predict==y_test)/float(len(y_test))
            perf_train[g]==np.sum(predict_train==y_train)/float(len(y_train))
            wei[g,:]=model.params
            g=g+1
        performance=np.mean(perf)
        performance_train=np.mean(perf_train)
        weights=np.mean(wei,axis=0)
        output={'performance':performance,'performance_train':performance_train,'weights':weights}
        return output

    def logregress_shuffled(self,n):
        perf=np.zeros((self.cv,n))
        perf_train=np.zeros((self.cv,n))
        wei=np.zeros((self.cv,n,len(self.feat[0])))
        for k in range(n):  
            clase_s=permutation(self.clase)
            skf=KFold(len(clase_s),self.cv) 
            g=0
            for train,test in skf: 
                X_train=self.feat[train]
                X_test=self.feat[test]
                y_train=clase_s[train]
                y_test=clase_s[test]
                lr=sm.Logit(endog=y_train,exog=X_train)
                model_pre=lr.fit(disp=False)
                model=lr.fit_regularized(start_params=model_pre.params,alpha=self.regularization,disp=False)
                predict=(model.predict(exog=X_test))>0.5
                predict_train=(model.predict(exog=X_train))>0.5
                perf[g,k]=np.sum(predict==y_test)/float(len(y_test))
                perf_train[g,k]==np.sum(predict_train==y_train)/float(len(y_train))
                wei[g,k,:]=model.params
                g=g+1
        performance_distr=np.mean(perf,axis=0)
        performance_train_distr=np.mean(perf_train,axis=0)
        weights_distr=np.mean(wei,axis=0)
        output={'performance_distr':performance_distr,'performance_train_distr':performance_train_distr,'weights_distr':weights_distr}
        return output

# ROC donde se le pasa solo las clases y las features, sin tiempo ni nada, del estilo del Logregress de arriba
class ROC_standard:
    'Solo se pasa una neurona'
    def __init__(self,feat,clase): 
        self.feat=feat
        self.feat=np.reshape(self.feat,len(self.feat))
        self.clase=clase
        self.clase_unique=np.unique(self.clase)
        if len(np.where(self.feat<0)[0])!=0:
            print 'z-score On, it should be removed'
                       
    def ROC(self):
        index_clase1=np.where(self.clase==self.clase_unique[0])[0] 
        index_clase2=np.where(self.clase==self.clase_unique[1])[0]   
# ponemos cada feature a la clase que le corresponde
        feature_1=self.feat[index_clase1]
        feature_2=self.feat[index_clase2]
        ma=max(max(feature_1),max(feature_2))
        delta_x=10.0
        resolution=delta_x*ma
        if resolution==0.0:
            perf=0.5
        else:
            hist1=np.histogram(feature_1,range=(0,ma),bins=resolution,density=True)[0]*(1.0/delta_x)
            hist2=np.histogram(feature_2,range=(0,ma),bins=resolution,density=True)[0]*(1.0/delta_x)
            p=[]
# Aqui hacemos la integral
            for l in range(len(hist1)):
                s=hist1[l]*(np.sum(hist2[(l+1):])) 
                s=s+hist1[l]*hist2[l]/2.0
                p.append(s)
            perf=np.sum(p)
        if np.isnan(perf)==True:
            print 'aqui hay nan'
        return perf
    
    def ROC_shuffled(self,n):
        perf_dist=[]
        for i in range(n):
            clase_s=permutation(self.clase)
# encontramos los indices a los que pertenece cada clase                
            index_clase1=np.where(clase_s==self.clase_unique[0])[0] 
            index_clase2=np.where(clase_s==self.clase_unique[1])[0]                     
# ponemos cada feature a la clase que le corresponde
            feature_1=self.feat[index_clase1]
            feature_2=self.feat[index_clase2]
            ma=max(max(feature_1),max(feature_2))
            delta_x=10.0
            resolution=delta_x*ma
            if resolution==0.0:
                perf=0.5
            else:
                hist1=np.histogram(feature_1,range=(0,ma),bins=resolution,density=True)[0]*(1.0/delta_x)
                hist2=np.histogram(feature_2,range=(0,ma),bins=resolution,density=True)[0]*(1.0/delta_x)
                p=[]
# Aqui hacemos la integral
                for l in range(len(hist1)):
                    s=hist1[l]*(np.sum(hist2[(l+1):])) 
                    s=s+hist1[l]*hist2[l]/2.0
                    p.append(s)
                perf=np.sum(p)
            perf_dist.append(perf)
            if np.isnan(perf)==True:
                print 'aqui hay sh nan'
        perf_dist=np.array(perf_dist)  
        return perf_dist

# ROC donde se le pasa solo las clases y las features, sin tiempo ni nada, del estilo del Logregress de arriba
class ROC_absolute:
    'Solo se pasa una neurona'
    def __init__(self,feat,clase): 
        self.feat=feat
        self.feat=np.reshape(self.feat,len(self.feat))
        self.clase=clase
        self.clase_unique=np.unique(self.clase)
        if len(np.where(self.feat<0)[0])!=0:
            print 'z-score On, it should be removed'
                       
    def ROC(self):
        index_clase1=np.where(self.clase==self.clase_unique[0])[0] 
        index_clase2=np.where(self.clase==self.clase_unique[1])[0]   
# ponemos cada feature a la clase que le corresponde
        feature_1=self.feat[index_clase1]
        feature_2=self.feat[index_clase2]
        ma=max(max(feature_1),max(feature_2))
        delta_x=10.0
        resolution=int(delta_x*ma)
        if resolution==0.0:
            perf=0.5
        else:
            hist1=np.histogram(feature_1,range=(0,ma),bins=resolution,density=True)[0]*(1.0/delta_x)
            hist2=np.histogram(feature_2,range=(0,ma),bins=resolution,density=True)[0]*(1.0/delta_x)
            p=[]
# Aqui hacemos la integral
            for l in range(len(hist1)):
                s=hist1[l]*(np.sum(hist2[(l+1):])) 
                s=s+hist1[l]*hist2[l]/2.0
                p.append(s)
            perf=np.sum(p)
        if perf>=0.5:
            perf=perf
        if perf<0.5:
            perf=1.0-perf
        if np.isnan(perf)==True:
            print 'aqui hay nan'
        return perf
    
    def ROC_shuffled(self,n):
        perf_dist=[]
        for i in range(n):
            clase_s=permutation(self.clase)
# encontramos los indices a los que pertenece cada clase                
            index_clase1=np.where(clase_s==self.clase_unique[0])[0] 
            index_clase2=np.where(clase_s==self.clase_unique[1])[0]                     
# ponemos cada feature a la clase que le corresponde
            feature_1=self.feat[index_clase1]
            feature_2=self.feat[index_clase2]
            ma=max(max(feature_1),max(feature_2))
            delta_x=10.0
            resolution=int(delta_x*ma)
            if resolution==0.0:
                perf=0.5
            else:
                hist1=np.histogram(feature_1,range=(0,ma),bins=resolution,density=True)[0]*(1.0/delta_x)
                hist2=np.histogram(feature_2,range=(0,ma),bins=resolution,density=True)[0]*(1.0/delta_x)
                p=[]
# Aqui hacemos la integral
                for l in range(len(hist1)):
                    s=hist1[l]*(np.sum(hist2[(l+1):])) 
                    s=s+hist1[l]*hist2[l]/2.0
                    p.append(s)
                perf=np.sum(p)
            if perf>=0.5:
                perf=perf
            if perf<0.5:
                perf=1.0-perf
            perf_dist.append(perf)
            if np.isnan(perf)==True:
                print 'aqui hay sh nan'
        perf_dist=np.array(perf_dist)  
        return perf_dist


    # Usamos el decodificador ROC con el sampling. Las variables de entrada son 2: vector con spike rate para cada trial y clase a al que pertenece cada trial. 
class ROC_montecarlo:
    'Solo se pasa una neurona'
    def __init__(self,feat,clase,resolution=int(10E4),shuff=False):
        self.feat=feat
        self.clase=clase 
        self.resolution=resolution
        
    def ROC(self):
        clase_unique=np.unique(self.clase)
        index_clase1=np.where(self.clase==clase_unique[0])[0] 
        index_clase2=np.where(self.clase==clase_unique[1])[0] 

        feature_1=self.feat[index_clase1]
        feature_2=self.feat[index_clase2]

        t=0
        for l in range(self.resolution):
            x=np.random.randint(0,len(feature_1))
            y=np.random.randint(0,len(feature_2))
            if feature_1[x]>feature_2[y]:
                t=t+1
            if feature_1[x]==feature_2[y]:
                t=t+0.5
        t=float(t)
        perf_pre=t/self.resolution

        return 1-perf_pre # cuidado con esto

class generalized_linear_model_no_cv:
    def __init__(self,noise_model,feat,target,regularization):
        self.feat=sm.add_constant(feat,prepend=False) # En los weights el ultimo valor es el del intercept
        self.clase=target 
        self.noise_model=noise_model
        if noise_model=='logistic_regression':
            self.clase[self.clase==-1]=0
        self.regularization=regularization
        self.exposure=0.15*np.ones(len(self.clase))

    def glm(self):
        if self.noise_model=='logistic_regression':
            lr=sm.GLM(endog=self.clase,exog=self.feat,family=sm.families.Binomial())
            model=lr.fit()
            performance=model.deviance
        if self.noise_model=='poisson_regression':
            lr=sm.Poisson(endog=self.clase,exog=self.feat,offset=self.exposure)
            if self.regularization==0.0:
                model=lr.fit(method='powell',disp=False)
                params=model.params
            else:
                model_pre=lr.fit(method='powell',disp=False)
                params_pre=model_pre.params
                model=lr.fit_regularized(start_params=params_pre,method='l1',disp=False,alpha=self.regularization,trim_mode='size',maxiter=100000)
            performance=model.prsquared
        if self.noise_model=='linear_regression':
            lr=sm.OLS(endog=self.clase,exog=self.feat)
            if self.regularization==0.0:
                model=lr.fit()
            else:
                model=lr.fit_regularized(alpha=self.regularization,L1_wt=0.0)
            performance=model.rsquared
        llh=model.llf
        aic=model.aic
        bic=model.bic
        weights=model.params
        output={'performance':performance,'weights':weights,'llh':llh,'aic':aic,'bic':bic}
        return output
        
    def glm_shuffled(self,n):
        performance=np.zeros(n)
        weights=np.zeros((n,len(self.feat[0])))
        k=0
        while k<n:
            clase_s=permutation(self.clase)
            try:
                if self.noise_model=='logistic_regression':
                    lr=sm.GLM(endog=clase_s,exog=self.feat,family=sm.families.Binomial())
                    model=lr.fit()
                    performance[k]=model.deviance
                if self.noise_model=='poisson_regression':
                    lr=sm.Poisson(endog=clase_s,exog=self.feat)
                    if self.regularization==0.0:
                        model=lr.fit(method='powell',disp=False)
                        params=model.params
                    else:
                        model_pre=lr.fit(method='powell',disp=False)
                        params_pre=model_pre.params
                        model=lr.fit_regularized(start_params=params_pre,method='l1',disp=False,alpha=self.regularization,trim_mode='size')
                    performance[k]=model.prsquared
                if self.noise_model=='linear_regression':
                    lr=sm.OLS(endog=clase_s,exog=self.feat)
                    if self.regularization==0.0:
                        model=lr.fit()
                    else:
                        model=lr.fit_regularized(alpha=self.regularization,L1_wt=0.0)
                    performance[k]=model.rsquared
                weights[k]=model.params
                k=k+1
            except:
                print 'except sh'
        output={'performance_distr':performance,'weights_distr':weights}
        return output

