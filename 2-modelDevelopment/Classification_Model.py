# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:13:00 2021

@author: Administrator
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import svm, linear_model
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc,confusion_matrix
from sklearn.model_selection import LeaveOneOut,KFold
from sklearn.svm import SVC,LinearSVC
from sklearn.feature_selection import SelectFdr, chi2,mutual_info_classif
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import scipy.stats as stats
from pandas import DataFrame as DF
from imblearn.over_sampling import SMOTE
import xlrd
import os
from comparision_auc_delong import delong_roc_test
import seaborn as sns


def confindence_interval_compute(y_pred, y_true):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
#        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        indices = rng.randint(0, len(y_pred)-1, len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_std = sorted_scores.std()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower,confidence_upper,confidence_std

def prediction_score(truth, predicted):
    TN, FP, FN, TP = confusion_matrix(truth, predicted, labels=[0,1]).ravel()
    print(TN, FP, FN, TP)
    ACC = (TP+TN)/(TN+FP+FN+TP)
    SEN = TP/(FN+TP)
    SPE = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    print('ACC:',ACC)
    print('Sensitivity:',SEN)
    print('Specifity:',SPE)
    print('PPV:',PPV)
    print('NPV:',NPV)
    OR = (TP*TN)/(FP*FN)
    print('OR:',OR)

def Find_Optimal_Cutoff(TPR, FPR, threshold):
   y = TPR - FPR
   Youden_index = np.argmax(y)  # Only the first occurrence is returned.
   optimal_threshold = threshold[Youden_index]
   point = [FPR[Youden_index], TPR[Youden_index]]
   return optimal_threshold, point

def calculate_net_benefit_model(thresh_group, y_pred_score, y_label):
    net_benefit_model = np.array([])
    for thresh in thresh_group:
        y_pred_label = y_pred_score > thresh
        tn, fp, fn, tp = confusion_matrix(y_label, y_pred_label).ravel()
        n = len(y_label)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_model = np.append(net_benefit_model, net_benefit)
    return net_benefit_model


def calculate_net_benefit_all(thresh_group, y_label):
    net_benefit_all = np.array([])
    tn, fp, fn, tp = confusion_matrix(y_label, y_label).ravel()
    total = tp + tn
    for thresh in thresh_group:
        net_benefit = (tp / total) - (tn / total) * (thresh / (1 - thresh))
        net_benefit_all = np.append(net_benefit_all, net_benefit)
    return net_benefit_all

    
if __name__ == '__main__':
    font = {'family' : 'Times New Roman',
 			'weight' : 'normal',
 			'size'   : 12,}
    plt.rc('font', **font)
    ## Tumor Feature   
    train_path1 = '../Results/TrainingData_Radiomics_Feature.csv'
    train_list1 = pd.read_csv(train_path1)
    tag = np.any(train_list1.isnull()== True)
    if tag:
    	#-- 中值补齐 --#
        train_path1 = train_list1.fillna(train_path1.median())
    pass
    train_PatientID1 = list(np.array(train_list1['Patient_ID']).astype(int))
    train_Feature1 = np.array(train_list1.values[:,:-2])
    train_FeatureName = np.array(list(train_list1.head(0))[:-2])
    
    train_data1 = xlrd.open_workbook('../data_list/TrainingDataset.xls')
    train_table1 = train_data1.sheets()[0]
    train_name1 = train_table1.col_values(0)[1:]
    train_IDs1 = np.array(train_table1.col_values(15)[1:]).astype(int)
    train_Class1 = np.array(train_table1.col_values(16)[1:]).astype(int)
    train_PFS1 = np.array(train_table1.col_values(18)[1:]).astype(int)
    train_PFS_Time1  = np.array(train_table1.col_values(23)[1:]).astype(int)
    train_sex1 = np.array(train_table1.col_values(1)[1:])
    train_sex1 = [1 if i=='male' else 0 for i in train_sex1]
    train_age1 = np.array(train_table1.col_values(2)[1:]).astype(int)
    train_size1 = np.array(train_table1.col_values(8)[1:]).astype(float)
    train_location1 = np.array(train_table1.col_values(11)[1:])
    train_location1 = [1 if i=='C' else 0 for i in train_location1]
    train_ind1 = [train_PatientID1.index(i) for i in train_IDs1]
    train_Feature1 = train_Feature1[train_ind1,:]
    train_Clinical1 = np.array([train_sex1, train_age1, train_size1, train_location1]).transpose(1,0)
    
    train_path2 = '../Results/trainSub_Radiomics_Feature.csv'
    train_list2 = pd.read_csv(train_path2)
    tag = np.any(train_list2.isnull()== True)
    if tag:
    	#-- 中值补齐 --#
        train_path2 = train_list2.fillna(train_path2.median())
    pass
    train_PatientID2 = list(np.array(train_list2['Patient_ID']).astype(int))
    train_Class2 = np.array(train_list2['Class']).astype(int)
    train_Feature2 = np.array(train_list2.values[:,:-2])
    # train_FeatureName = np.array(list(train_list2.head(0))[:-2])
    
    train_data2 = xlrd.open_workbook('../data_list/PatientList.xls')
    train_table2 = train_data2.sheets()[0]
    train_name2 = train_table2.col_values(3)[1:]
    train_IDs2 = train_table2.col_values(0)[1:]
    train_IDs2 = [int(i) for i in train_IDs2]
    train_PFS2 = np.array(train_table2.col_values(14)[1:]).astype(int)
    train_PFS_Time2 = np.array(train_table2.col_values(17)[1:]).astype(int)
    train_sex2 = np.array(train_table2.col_values(4)[1:])
    train_sex2 = [1 if i=='男' else 0 for i in train_sex2]
    train_age2 = np.array(train_table2.col_values(5)[1:]).astype(int)
    train_size2 = np.array(train_table2.col_values(9)[1:]).astype(float)
    train_location2 = np.array(train_table2.col_values(11)[1:])
    train_location2 = [1 if i=='C' else 0 for i in train_location2]
    train_ind2 = [train_PatientID2.index(i) for i in train_IDs2]
    train_Feature2 = train_Feature2[train_ind2,:]
    train_Class2 = train_Class2[train_ind2]
    train_Clinical2 = np.array([train_sex2, train_age2, train_size2, train_location2]).transpose(1,0)
    
    train_Feature = np.vstack((train_Feature1, train_Feature2))
    train_PFS = np.hstack((train_PFS1, train_PFS2))
    train_PFS_Time = np.hstack((train_PFS_Time1, train_PFS_Time2))
    train_Clinical = np.vstack((train_Clinical1, train_Clinical2))
    train_Class = np.hstack((train_Class1, train_Class2))
    
    train_name = train_name1+train_name2

    train_save = {}
    train_save['Name'] = train_name
    train_save['ID'] = np.hstack((train_IDs1,train_IDs2))
    train_save['Sex'] = train_Clinical[:,0]
    train_save['Age'] = train_Clinical[:,1]
    train_save['Size'] = train_Clinical[:,2]
    train_save['Loaction'] = train_Clinical[:,3]
    train_save['DFS'] = train_PFS
    train_save['DFS_Time'] = train_PFS_Time
    train_save['LN'] = train_Class


    
    test_path = '../Results/TestingData_Radiomics_Feature.csv'
    TestData = pd.read_csv(test_path)
    test_tag = np.any(TestData.isnull()== True)
    if test_tag:
        TestData = TestData.fillna(TestData.median())
    pass    
    test_Feature = np.array(TestData.values[:,:-2])
    test_PatientID = list(np.array(TestData['Patient_ID']).astype(int))
    
    test_data = xlrd.open_workbook('../data_list/TestingDataset.xls')
    test_table = test_data.sheets()[0]
    test_name = test_table.col_values(1)[1:]
    test_IDs = np.array(test_table.col_values(16)[1:]).astype(int)
    test_Class = np.array(test_table.col_values(17)[1:]).astype(int)
    test_PFS = np.array(test_table.col_values(19)[1:]).astype(int)
    test_PFS_Time = np.array(test_table.col_values(24)[1:]).astype(int)
    test_sex = np.array(test_table.col_values(2)[1:])
    test_sex = [1 if i=='男' else 0 for i in test_sex]
    test_age = np.array(test_table.col_values(3)[1:]).astype(int)
    test_size = np.array(test_table.col_values(10)[1:]).astype(float)
    test_loacation = np.array(test_table.col_values(12)[1:])
    test_loacation = [1 if i=='C' else 0 for i in test_loacation]
    test_ind = [test_PatientID.index(i) for i in test_IDs]
    test_Feature = test_Feature[test_ind,:]
    test_Clinical = np.array([test_sex, test_age, test_size, test_loacation]).transpose(1,0)
    
    test_save = {}
    test_save['Name'] = test_name
    test_save['ID'] = test_IDs
    test_save['Sex'] = test_Clinical[:,0]
    test_save['Age'] = test_Clinical[:,1]
    test_save['Size'] = test_Clinical[:,2]
    test_save['Loaction'] = test_Clinical[:,3]
    test_save['DFS'] = test_PFS
    test_save['DFS_Time'] = test_PFS_Time
    test_save['LN'] = test_Class
    
    TJCH_path = '../Results/TJCH_Radiomics_Feature.csv'
    TJCHData = pd.read_csv(TJCH_path)
    TJCH_tag = np.any(TJCHData.isnull()== True)
    if TJCH_tag:
        TJCHData = TJCHData.fillna(TJCHData.median())
    pass 
    TJCH_PatientID = TJCHData['Patient_ID']
    # y_TJCH = np.array(TJCHData['Class'])
    TJCH_PatientName = [''.join([i for i in string if not i.isdigit()]) for string in TJCH_PatientID]
    TJCH_PatientName = [i.lower() for i in TJCH_PatientName]
    
    data = xlrd.open_workbook('../data_list/TJCH.xls')
    table = data.sheets()[0]
    List_all = table.col_values(0)[1:]
    TJCH_OS_Time = np.array(table.col_values(-1)[1:])
    TJCH_DFS_Time = np.array(table.col_values(-2)[1:])
    TJCH_LLRFS_Time = np.array(table.col_values(-3)[1:])
    TJCH_OS = np.array(table.col_values(-4)[1:])
    TJCH_DFS = np.array(table.col_values(14)[1:])
    TJCH_LLRFS = np.array(table.col_values(-5)[1:])
    TJCH_HisType = np.array(table.col_values(13)[1:])
    TJCH_sex = np.array(table.col_values(2)[1:])
    TJCH_sex = [1 if i=='male' else 0 for i in TJCH_sex]
    TJCH_age = np.array(table.col_values(1)[1:]).astype(int)
    TJCH_size = np.array(table.col_values(6)[1:]).astype(float)
    TJCH_location = np.array(table.col_values(5)[1:]).astype(int)
    ind_TJCH = [TJCH_PatientName.index(i) for i in List_all]
    y_TJCH = TJCH_DFS
    TJCH_Feature = TJCHData.values[ind_TJCH,:-2]
    TJCH_PatientID = np.array(TJCH_PatientID)[ind_TJCH]
    
    VD2_path = '../Results/VD2_Radiomics_Feature.csv'
    VD2Data = pd.read_csv(VD2_path)
    VD2_tag = np.any(VD2Data.isnull()== True)
    if VD2_tag:
        VD2Data = VD2Data.fillna(VD2Data.median())
    pass    
    VD2_Feature = np.array(VD2Data.values[:,:-2])
    VD2_PatientID = list(np.array(VD2Data['Patient_ID']).astype(int))

    VD2_data = xlrd.open_workbook('../data_list/SBRT-VD2.xls')
    VD2_table = VD2_data.sheets()[0]
    VD2_IDs = list(VD2_table.col_values(11)[1:])
    VD2_name = VD2_table.col_values(0)[1:]
    VD2_ind = [VD2_IDs.index(str(i)) for i in VD2_PatientID]
    VD2_DFS = np.array(VD2_table.col_values(18)[1:])[VD2_ind].astype(int)
    VD2_DFS_Time = np.array(VD2_table.col_values(-2)[1:])[VD2_ind].astype(int)
    VD2_LLRFS = np.array(VD2_table.col_values(16)[1:])[VD2_ind].astype(int)
    VD2_LLRFS_Time = np.array(VD2_table.col_values(-3)[1:])[VD2_ind].astype(int)
    VD2_OS = np.array(VD2_table.col_values(21)[1:])[VD2_ind].astype(int)
    VD2_OS_Time = np.array(VD2_table.col_values(-1)[1:])[VD2_ind].astype(int)
    VD2_sex = np.array(VD2_table.col_values(2)[1:])[VD2_ind]
    VD2_sex = [1 if i=='男' else 0 for i in VD2_sex]
    VD2_age = np.array(VD2_table.col_values(1)[1:])[VD2_ind].astype(int)
    VD2_size = np.array(VD2_table.col_values(3)[1:])[VD2_ind].astype(float)
    VD2_location = np.array(VD2_table.col_values(4)[1:])[VD2_ind]
    VD2_location = [1 if i=='C' else 0 for i in VD2_location]
    VD2_name = np.array(VD2_name)[VD2_ind]

    PER_Feature = np.vstack((TJCH_Feature, VD2_Feature))
    PER_sex = np.hstack((np.array(TJCH_sex), np.array(VD2_sex)))
    PER_age = np.hstack((TJCH_age, VD2_age))
    PER_size = np.hstack((TJCH_size, VD2_size))
    PER_location = np.hstack((np.array(TJCH_location), np.array(VD2_location)))
    PER_DFS = np.hstack((TJCH_DFS, VD2_DFS))
    PER_DFS_Time = np.hstack((TJCH_DFS_Time, VD2_DFS_Time))
    PER_LLRFS = np.hstack((TJCH_LLRFS, VD2_LLRFS))
    PER_LLRFS_Time = np.hstack((TJCH_LLRFS_Time, VD2_LLRFS_Time))
    PER_OS = np.hstack((TJCH_OS, VD2_OS))
    PER_OS_Time = np.hstack((TJCH_OS_Time, VD2_OS_Time))
    PER_DFS = ((PER_DFS+PER_OS)>0).astype(int)
    PER_LLRFS = ((PER_LLRFS+PER_OS)>0).astype(int)
    PER_Clinical = np.array([PER_sex, PER_age, PER_size, PER_location]).transpose(1,0)
    PER_name = np.hstack((List_all, VD2_name))
    PER_IDs = np.hstack((List_all, VD2_PatientID))
    
    
    PER_save = {}
    PER_save['Name'] = PER_name
    PER_save['ID'] = PER_IDs
    PER_save['Sex'] = PER_Clinical[:,0]
    PER_save['Age'] = PER_Clinical[:,1]
    PER_save['Size'] = PER_Clinical[:,2]
    PER_save['Loaction'] = PER_Clinical[:,3]
    PER_save['PFS'] = PER_DFS
    PER_save['PFS_Time'] = PER_DFS_Time
    PER_save['LLRFS'] = PER_LLRFS
    PER_save['LLRFS_Time'] = PER_LLRFS_Time
    PER_save['OS'] = PER_OS
    PER_save['OS_Time'] = PER_OS_Time


    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    x_train_Rad = scaler.fit_transform(np.array(train_Feature))
    x_test_Rad = scaler.transform(test_Feature)
    x_PER_Rad = scaler.transform(PER_Feature)
    FDR_selector = SelectFdr().fit(x_train_Rad, train_Class)
    x_train_Rad = FDR_selector.transform(x_train_Rad)
    x_test_Rad = FDR_selector.transform(x_test_Rad)
    x_PER_Rad = FDR_selector.transform(x_PER_Rad)

    estimator_Rad = SVC(kernel="linear",random_state=0)
    selector_Img = RFE(estimator_Rad, n_features_to_select=8, step=1)
    # selector_Img = KernelPCA(n_components=i,random_state=0)
    train_Rad = selector_Img.fit_transform(x_train_Rad,train_Class)
    PER_Rad = selector_Img.transform(x_PER_Rad)
    test_Rad = selector_Img.transform(x_test_Rad)
    
    indices = list(np.where(selector_Img.support_==True)[0])
    SelectedFeatures = np.array(train_FeatureName)[indices]
    print(SelectedFeatures)
    selected_feature = x_train_Rad[:,indices]
    feature_names = ['F'+str(i) for i in range(1,9)]
    Class_Type = [i if i==0  else 'N+' for i in train_Class]
    Class_Type = [i if i=='N+' else 'N-' for i in Class_Type]
    selectedFeature = {}
    selectedFeature['FeatureName'] = np.ravel([[i]*len(selected_feature) for i in feature_names])
    selectedFeature['Feature'] = np.ravel([selected_feature[:,i] for i in range(selected_feature.shape[1])])
    selectedFeature['Class_Type'] = np.ravel([Class_Type for i in range(selected_feature.shape[1])])
    selectedFeature = pd.DataFrame.from_dict(selectedFeature)
    
    font = {'family' : 'Arial',
            'weight' :  'medium',
            'size'   : 10,}
    plt.rc('font', **font)
    plt.figure(figsize=(6,4))
    sns.boxenplot(x="FeatureName", y="Feature",hue='Class_Type',data=selectedFeature,
                      palette='Set2')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(title='',edgecolor='k',loc='upper right')
    plt.subplots_adjust(top=0.995,bottom=0.06,left=0.045,right=0.995,hspace=0,wspace=0)
    
    # x_Rad, y_Rad = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(train_Rad, train_Class)

    clf_Rad = svm.SVC(kernel="rbf", probability=True, random_state=0)
    clf_Rad.fit(train_Rad, train_Class)
    train_prob_Rad = clf_Rad.predict_proba(train_Rad)[:,1]
    pred_label_train_Rad = clf_Rad.predict(train_Rad)
    pred_label_train_Rad = np.array(pred_label_train_Rad).astype(int)
    fpr_train_Rad,tpr_train_Rad,threshold_train_Rad = roc_curve(train_Class, np.array(train_prob_Rad)) ###计算真正率和假正率
    auc_score_train_Rad = auc(fpr_train_Rad,tpr_train_Rad)
    auc_l_train_Rad, auc_h_train_Rad, auc_std_train_Rad = confindence_interval_compute(np.array(train_prob_Rad), train_Class)
    print('Training Dataset AUC:%.2f+/-%.2f'%(auc_score_train_Rad,auc_std_train_Rad),'  95%% CI:[%.2f,%.2f]'%(auc_l_train_Rad, auc_h_train_Rad))
    print('Training Dataset ACC:%.2f%%'%(accuracy_score(train_Class,pred_label_train_Rad)*100)) 
    prediction_score(train_Class, pred_label_train_Rad)
    print('-----------------------------------------------')

    
    test_prob_Rad = clf_Rad.predict_proba(test_Rad)[:,1]
    pred_label_Rad = clf_Rad.predict(test_Rad)
    pred_label_Rad = np.array(pred_label_Rad).astype(int)
    fpr_Rad,tpr_Rad,threshold_Rad = roc_curve(test_Class, np.array(test_prob_Rad)) ###计算真正率和假正率
    auc_score_Rad = auc(fpr_Rad,tpr_Rad)
    auc_l_Rad, auc_h_Rad, auc_std_Rad = confindence_interval_compute(np.array(test_prob_Rad), test_Class)
    print('Testing Dataset AUC:%.2f+/-%.2f'%(auc_score_Rad,auc_std_Rad),'  95%% CI:[%.2f,%.2f]'%(auc_l_Rad,auc_h_Rad))
    print('Testing Dataset  ACC:%.2f%%'%(accuracy_score(test_Class,pred_label_Rad)*100)) 
    # TN, FP, FN, TP = confusion_matrix(test_Class, pred_label_Rad, labels=[0,1]).ravel()
    # print(TN, FP, FN, TP)
    prediction_score(test_Class, test_prob_Rad>=0.5)
    print('-----------------------------------------------')
            
    train_ind_pos = [i for i in range(len(train_Class)) if train_Class[i]==1]
    train_ind_neg = [i for i in range(len(train_Class)) if train_Class[i]==0]
    t_train = stats.levene(np.array(train_prob_Rad)[train_ind_pos], np.array(train_prob_Rad)[train_ind_neg],center='median')
    #如果p不显著，方差不齐，如果p显著，方差齐
    if t_train[1]<0.05:
        t_train = stats.ttest_ind(np.array(train_prob_Rad)[train_ind_pos], np.array(train_prob_Rad)[train_ind_neg], equal_var=True)#True齐，False不齐
    else:
        t_train = stats.ttest_ind(np.array(train_prob_Rad)[train_ind_pos], np.array(train_prob_Rad)[train_ind_neg], equal_var=False)#True齐，False不齐  
    print('train: t=%6.4f,p=%6.8f'%t_train)#方差齐性检验
    
    test_ind_pos = [i for i in range(len(test_Class)) if (test_Class)[i]==1]
    test_ind_neg = [i for i in range(len(test_Class)) if (test_Class)[i]==0]
    t_test = stats.levene(np.array(test_prob_Rad)[test_ind_pos], np.array(test_prob_Rad)[test_ind_neg],center='median')
    #如果p不显著，方差不齐，如果p显著，方差齐
    if t_test[1]<0.05:
        t_test = stats.ttest_ind(np.array(test_prob_Rad)[test_ind_pos], np.array(test_prob_Rad)[test_ind_neg], equal_var=True)#True齐，False不齐
    else:
        t_test = stats.ttest_ind(np.array(test_prob_Rad)[test_ind_pos], np.array(test_prob_Rad)[test_ind_neg], equal_var=False)#True齐，False不齐  
    print('test: t=%6.4f,p=%6.8f'%t_test)#方差齐性检验
    
    prob = [train_prob_Rad, test_prob_Rad]
    dataset = ['Training Cohort', 'Validation Cohort']
    train_class_type = [i if i==0 else 'N+' for i in train_Class]
    train_class_type = [i if i=='N+' else 'N-' for i in train_class_type]
    test_class_type = [i if i==0 else 'N+' for i in test_Class]
    test_class_type = [i if i=='N+' else 'N-' for i in test_class_type]
    class_type = [train_class_type, test_class_type]
    
    Rad_Score = {}
    Rad_Score['Dataset'] = np.ravel([dataset[i] for i in range(len(dataset)) for j in range(len(prob[i]))])
    Rad_Score['Score'] = np.ravel([prob[i][j] for i in range(len(dataset)) for j in range(len(prob[i]))])
    Rad_Score['Class'] = np.ravel([class_type[i][j] for i in range(len(dataset)) for j in range(len(class_type[i]))])
    plt.figure(figsize=(5,5))
    sns.violinplot(x="Dataset", y="Score", hue="Class", data=Rad_Score, palette="muted", split=True, inner='quartile')
    plt.ylabel('Prediction Score')
    plt.legend(loc="best",edgecolor='k',fontsize=10,fancybox=False)
    plt.title('Radiomics Model')
    plt.text(-0.4,0.1,'P=%.1e'%(t_train[1]),bbox=dict(boxstyle="square",
    ec=(1., 0.5, 0.5),  fc=(1., 0.8, 0.8),))
    plt.text(0.6,0.1,'P=%.1e'%(t_test[1]),bbox=dict(boxstyle="square",
    ec=(1., 0.52, 0.5),  fc=(1., 0.8, 0.8),))
    plt.subplots_adjust(top=0.99,bottom=0.05,left=0.12,right=0.99,hspace=0,wspace=0)
    
    ## Clinical Model
    scaler = StandardScaler()
    train_Clinical = scaler.fit_transform(np.array(train_Clinical))
    test_Clinical = scaler.transform(test_Clinical)
    PER_Clinical = scaler.transform(PER_Clinical)
    # FDR_selector = SelectFdr().fit(train_Clinical, train_Class)
    # train_Clinical = FDR_selector.transform(train_Clinical)
    # test_Clinical = FDR_selector.transform(test_Clinical)
    # PER_Clinical = FDR_selector.transform(PER_Clinical)
    # x_Clinical, y_Clinical = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(train_Clinical, train_Class)

    clf_Clinical = svm.SVC(kernel="rbf", probability=True, random_state=0)

    clf_Clinical.fit(train_Clinical, train_Class)
    train_prob_Clinical = clf_Clinical.predict_proba(train_Clinical)[:,1]
    pred_label_train_Clinical = clf_Clinical.predict(train_Clinical)
    pred_label_train_Clinical = np.array(pred_label_train_Clinical).astype(int)
    fpr_train_Clinical,tpr_train_Clinical,threshold_train_Clinical = roc_curve(train_Class, np.array(train_prob_Clinical)) ###计算真正率和假正率
    auc_score_train_Clinical = auc(fpr_train_Clinical,tpr_train_Clinical)
    auc_l_train_Clinical, auc_h_train_Clinical, auc_std_train_Clinical = confindence_interval_compute(np.array(train_prob_Clinical), train_Class)
    print('Training Dataset AUC:%.2f+/-%.2f'%(auc_score_train_Clinical,auc_std_train_Clinical),'  95%% CI:[%.2f,%.2f]'%(auc_l_train_Clinical, auc_h_train_Clinical))
    print('Training Dataset ACC:%.2f%%'%(accuracy_score(train_Class,pred_label_train_Clinical)*100)) 
    prediction_score(train_Class, pred_label_train_Clinical)
    print('-----------------------------------------------')

    test_prob_Clinical = clf_Clinical.predict_proba(test_Clinical)[:,1]
    pred_label_Clinical = clf_Clinical.predict(test_Clinical)
    pred_label_Clinical = np.array(pred_label_Clinical).astype(int)
    fpr_Clinical,tpr_Clinical,threshold_Clinical = roc_curve(test_Class, np.array(test_prob_Clinical)) ###计算真正率和假正率
    auc_score_Clinical = auc(fpr_Clinical,tpr_Clinical)
    auc_l_Clinical, auc_h_Clinical, auc_std_Clinical = confindence_interval_compute(np.array(test_prob_Clinical), test_Class)
    print('Testing Dataset AUC:%.2f+/-%.2f'%(auc_score_Clinical,auc_std_Clinical),'  95%% CI:[%.2f,%.2f]'%(auc_l_Clinical,auc_h_Clinical))
    print('Testing Dataset  ACC:%.2f%%'%(accuracy_score(test_Class,pred_label_Clinical)*100)) 
    # TN, FP, FN, TP = confusion_matrix(test_Class, pred_label_Clinical, labels=[0,1]).ravel()
    # print(TN, FP, FN, TP)
    prediction_score(test_Class, test_prob_Clinical>=0.5)
    print('-----------------------------------------------')
    
    train_ind_pos = [i for i in range(len(train_Class)) if train_Class[i]==1]
    train_ind_neg = [i for i in range(len(train_Class)) if train_Class[i]==0]
    t_train = stats.levene(np.array(train_prob_Clinical)[train_ind_pos], np.array(train_prob_Clinical)[train_ind_neg],center='median')
    #如果p不显著，方差不齐，如果p显著，方差齐
    if t_train[1]<0.05:
        t_train = stats.ttest_ind(np.array(train_prob_Clinical)[train_ind_pos], np.array(train_prob_Clinical)[train_ind_neg], equal_var=True)#True齐，False不齐
    else:
        t_train = stats.ttest_ind(np.array(train_prob_Clinical)[train_ind_pos], np.array(train_prob_Clinical)[train_ind_neg], equal_var=False)#True齐，False不齐  
    print('train: t=%6.4f,p=%6.8f'%t_train)#方差齐性检验
    
    test_ind_pos = [i for i in range(len(test_Class)) if (test_Class)[i]==1]
    test_ind_neg = [i for i in range(len(test_Class)) if (test_Class)[i]==0]
    t_test = stats.levene(np.array(test_prob_Clinical)[test_ind_pos], np.array(test_prob_Clinical)[test_ind_neg],center='median')
    #如果p不显著，方差不齐，如果p显著，方差齐
    if t_test[1]<0.05:
        t_test = stats.ttest_ind(np.array(test_prob_Clinical)[test_ind_pos], np.array(test_prob_Clinical)[test_ind_neg], equal_var=True)#True齐，False不齐
    else:
        t_test = stats.ttest_ind(np.array(test_prob_Clinical)[test_ind_pos], np.array(test_prob_Clinical)[test_ind_neg], equal_var=False)#True齐，False不齐  
    print('test: t=%6.4f,p=%6.8f'%t_test)#方差齐性检验
    
    prob = [train_prob_Clinical, test_prob_Clinical]
    dataset = ['Training Cohort', 'Validation Cohort']
    train_class_type = [i if i==0 else 'N+' for i in train_Class]
    train_class_type = [i if i=='N+' else 'N-' for i in train_class_type]
    test_class_type = [i if i==0 else 'N+' for i in test_Class]
    test_class_type = [i if i=='N+' else 'N-' for i in test_class_type]
    class_type = [train_class_type, test_class_type]
    
    Cli_Score = {}
    Cli_Score['Dataset'] = np.ravel([dataset[i] for i in range(len(dataset)) for j in range(len(prob[i]))])
    Cli_Score['Score'] = np.ravel([prob[i][j] for i in range(len(dataset)) for j in range(len(prob[i]))])
    Cli_Score['Class'] = np.ravel([class_type[i][j] for i in range(len(dataset)) for j in range(len(class_type[i]))])
    plt.figure(figsize=(5,5))
    sns.violinplot(x="Dataset", y="Score", hue="Class", data=Cli_Score, palette="muted", split=True, inner='quartile')
    plt.ylabel('Prediction Score')
    plt.legend(loc="best",edgecolor='k',fontsize=10,fancybox=False)
    plt.title('Clinical Model')
    plt.text(-0.4,0.425,'P=%.1e'%(t_train[1]),bbox=dict(boxstyle="square",
    ec=(1., 0.5, 0.5),  fc=(1., 0.8, 0.8),))
    plt.text(0.6,0.425,'P=%.1e'%(t_test[1]),bbox=dict(boxstyle="square",
    ec=(1., 0.52, 0.5),  fc=(1., 0.8, 0.8),))
    plt.subplots_adjust(top=0.99,bottom=0.05,left=0.15,right=0.99,hspace=0,wspace=0)
    
    
    # scales = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # for scale in scales:
    #     train_prob_fusion = scale*np.array(train_prob_Rad)+(1-scale)*np.array(train_prob_Clinical)
    #     train_auc_value = roc_auc_score(np.array(train_Class),train_prob_fusion)
    #     train_auc_fl, train_auc_fh, train_auc_fstd = confindence_interval_compute(np.array(train_prob_fusion), train_Class)
    #     print('Fusion Scale',scale,'AUC:%.2f'%train_auc_value,'+/-%.2f'%train_auc_fstd,
    #           '  95% CI:[','%.2f,'%train_auc_fl,'%.2f'%train_auc_fh,']')      
    # train_Fusion = np.zeros([len(train_prob_Rad),2])
    # train_Fusion[:,0] = np.array(train_prob_Rad)
    # train_Fusion[:,1] = np.array(train_prob_Clinical)
    # train_Fusion_min = train_Fusion.min(1)
    # train_Fusion_max = train_Fusion.max(1)
    
    
    # train_auc_min = roc_auc_score(np.array(train_Class),train_Fusion_min)
    # train_auc_fl_min, train_auc_fh_min, train_auc_fstd_min = confindence_interval_compute(np.array(train_Fusion_min), train_Class)
    # print('Min Fusion AUC:%.2f'%train_auc_min,'+/-%.2f'%train_auc_fstd_min,'  95% CI:[','%.2f,'%train_auc_fl_min,'%.2f'%train_auc_fh_min,']')
    
    # train_auc_max = roc_auc_score(np.array(train_Class),train_Fusion_max)
    # train_auc_fl_max, train_auc_fh_max,train_auc_fstd_max = confindence_interval_compute(np.array(train_Fusion_max), train_Class)
    # print('Max Fusion AUC:%.2f'%train_auc_max,'+/-%.2f'%train_auc_fstd_max,'  95% CI:[','%.2f,'%train_auc_fl_max,'%.2f'%train_auc_fh_max,']')
    # print('-----------------------------------------------')
    
    # for scale in scales:
    #     test_prob_fusion = scale*np.array(test_prob_Rad)+(1-scale)*np.array(test_prob_Clinical)
    #     test_auc_value = roc_auc_score(np.array(test_Class),test_prob_fusion)
    #     test_auc_fl, test_auc_fh, test_auc_fstd = confindence_interval_compute(np.array(test_prob_fusion), test_Class)
    #     print('Fusion Scale',scale,'AUC:%.2f'%test_auc_value,'+/-%.2f'%test_auc_fstd,
    #           '  95% CI:[','%.2f,'%test_auc_fl,'%.2f'%test_auc_fh,']')      
    # test_Fusion = np.zeros([len(test_prob_Rad),2])
    # test_Fusion[:,0] = np.array(test_prob_Rad)
    # test_Fusion[:,1] = np.array(test_prob_Clinical)
    # test_Fusion_min = test_Fusion.min(1)
    # test_Fusion_max = test_Fusion.max(1)
    
    # test_auc_min = roc_auc_score(np.array(test_Class),test_Fusion_min)
    # test_auc_fl_min, test_auc_fh_min, test_auc_fstd_min = confindence_interval_compute(np.array(test_Fusion_min), test_Class)
    # print('Min Fusion AUC:%.2f'%test_auc_min,'+/-%.2f'%test_auc_fstd_min,'  95% CI:[','%.2f,'%test_auc_fl_min,'%.2f'%test_auc_fh_min,']')
    
    # test_auc_max = roc_auc_score(np.array(test_Class),test_Fusion_max)
    # test_auc_fl_max, test_auc_fh_max,test_auc_fstd_max = confindence_interval_compute(np.array(test_Fusion_max), test_Class)
    # print('Max Fusion AUC:%.2f'%test_auc_max,'+/-%.2f'%test_auc_fstd_max,'  95% CI:[','%.2f,'%test_auc_fl_max,'%.2f'%test_auc_fh_max,']')

    scale = 0.4
    train_prob_fusion = scale*np.array(train_prob_Rad)+(1-scale)*np.array(train_prob_Clinical)
    fpr_train_Fusion,tpr_train_Fusion,threshold_train_Fusion = roc_curve(train_Class, np.array(train_prob_fusion))
    train_auc_value = auc(fpr_train_Fusion,tpr_train_Fusion)
    train_auc_fl, train_auc_fh, train_auc_fstd = confindence_interval_compute(np.array(train_prob_fusion), train_Class)
    print('Fusion Scale',scale,'Training Dataset AUC:%.2f'%train_auc_value,'+/-%.2f'%train_auc_fstd,
          '  95% CI:[','%.2f,'%train_auc_fl,'%.2f'%train_auc_fh,']')  
    print('Training Dataset ACC:%.2f%%'%(accuracy_score(train_Class,train_prob_fusion>0.5)*100)) 
    prediction_score(train_Class, train_prob_fusion>0.5)
    print('-----------------------------------------------')
    # scale = 0.1
    test_prob_fusion = scale*np.array(test_prob_Rad)+(1-scale)*np.array(test_prob_Clinical)
    fpr_test_Fusion,tpr_test_Fusion,threshold_test_Fusion = roc_curve(test_Class, np.array(test_prob_fusion))
    test_auc_value = auc(fpr_test_Fusion,tpr_test_Fusion)
    test_auc_fl, test_auc_fh, test_auc_fstd = confindence_interval_compute(np.array(test_prob_fusion), test_Class)
    print('Testing Dataset AUC:%.2f'%test_auc_value,'+/-%.2f'%test_auc_fstd,
          '  95% CI:[','%.2f,'%test_auc_fl,'%.2f'%test_auc_fh,']')   
    print('Testing Dataset  ACC:%.2f%%'%(accuracy_score(test_Class,test_prob_fusion>0.5)*100))
    prediction_score(test_Class, test_prob_fusion>=0.5)
    print('-----------------------------------------------')
    
    train_save['RadScore'] = test_prob_Rad
    train_save['CliScore'] = train_prob_Clinical
    train_save['FusionScore'] = train_prob_fusion
    df_train = DF(train_save)
    df_train.to_csv(os.path.join("../Results",'Result_TrainDataset.csv'),encoding="utf_8_sig")


    test_save['RadScore'] = test_prob_Rad
    test_save['CliScore'] = test_prob_Clinical
    test_save['FusionScore'] = test_prob_fusion
    df_test = DF(test_save)
    df_test.to_csv(os.path.join("../Results",'Result_TestDataset.csv'),encoding="utf_8_sig")

    PER_prob_Rad = clf_Rad.predict_proba(PER_Rad)[:,1]
    PER_prob_Clinical = clf_Clinical.predict_proba(PER_Clinical)[:,1]
    PER_prob_fusion = scale*np.array(PER_prob_Rad)+(1-scale)*np.array(PER_prob_Clinical)
    PER_save['RadScore'] = PER_prob_Rad
    PER_save['CliScore'] = PER_prob_Clinical
    PER_save['FusionScore'] = PER_prob_fusion
    df_PER = DF(PER_save)
    df_PER.to_csv(os.path.join("../Results",'Result_SBRTDataset.csv'),encoding="utf_8_sig")

    ## Comparision
    print('Training Cohort')
    P = delong_roc_test(train_Class, train_prob_fusion, train_prob_Rad)
    print('Fusion VS Radiomics P:%.3f'%P[0][0])
    P = delong_roc_test(train_Class, train_prob_fusion, train_prob_Clinical)
    print('Fusion VS Clinical P:%.3f'%P[0][0])
    P = delong_roc_test(train_Class, train_prob_Clinical, train_prob_Rad)
    print('Clinical VS Radiomics P:%.3f'%P[0][0])
    print('-----------------------------------------------')
    print('Testing Cohort')
    P = delong_roc_test(test_Class, test_prob_fusion, test_prob_Rad)
    print('Fusion VS Radiomics P:%.3f'%P[0][0])
    P = delong_roc_test(test_Class, test_prob_fusion, test_prob_Clinical)
    print('Fusion VS Clinical P:%.3f'%P[0][0])
    P = delong_roc_test(test_Class, test_prob_Clinical, test_prob_Rad)
    print('Clinical VS Radiomics P:%.3f'%P[0][0])
    print('-----------------------------------------------')
    
    
    lw = 1.5
    font = {'family' : 'Arial',
            'weight' :  'medium',
            'size'   : 10,}
    plt.rc('font', **font)
    
    
    train_ind_pos = [i for i in range(len(train_Class)) if train_Class[i]==1]
    train_ind_neg = [i for i in range(len(train_Class)) if train_Class[i]==0]
    t_train = stats.levene(np.array(train_prob_fusion)[train_ind_pos], np.array(train_prob_fusion)[train_ind_neg],center='median')
    #如果p不显著，方差不齐，如果p显著，方差齐
    if t_train[1]<0.05:
        t_train = stats.ttest_ind(np.array(train_prob_fusion)[train_ind_pos], np.array(train_prob_fusion)[train_ind_neg], equal_var=True)#True齐，False不齐
    else:
        t_train = stats.ttest_ind(np.array(train_prob_fusion)[train_ind_pos], np.array(train_prob_fusion)[train_ind_neg], equal_var=False)#True齐，False不齐  
    print('train: t=%6.4f,p=%6.8f'%t_train)#方差齐性检验
    
    test_ind_pos = [i for i in range(len(test_Class)) if test_Class[i]==1]
    test_ind_neg = [i for i in range(len(test_Class)) if test_Class[i]==0]
    t_test = stats.levene(np.array(test_prob_fusion)[test_ind_pos], np.array(test_prob_fusion)[test_ind_neg],center='median')
    #如果p不显著，方差不齐，如果p显著，方差齐
    if t_test[1]<0.05:
        t_test = stats.ttest_ind(np.array(test_prob_fusion)[test_ind_pos], np.array(test_prob_fusion)[test_ind_neg], equal_var=True)#True齐，False不齐
    else:
        t_test = stats.ttest_ind(np.array(test_prob_fusion)[test_ind_pos], np.array(test_prob_fusion)[test_ind_neg], equal_var=False)#True齐，False不齐  
    print('test: t=%6.4f,p=%6.8f'%t_test)#方差齐性检验
    
    prob = [train_prob_fusion, test_prob_fusion]
    dataset = ['Training Cohort', 'Validation Cohort']
    train_class_type = [i if i==0 else 'N+' for i in train_Class]
    train_class_type = [i if i=='N+' else 'N-' for i in train_class_type]
    test_class_type = [i if i==0 else 'N+' for i in test_Class]
    test_class_type = [i if i=='N+' else 'N-' for i in test_class_type]
    class_type = [train_class_type, test_class_type]
    
    Rad_Score = {}
    Rad_Score['Dataset'] = np.ravel([dataset[i] for i in range(len(dataset)) for j in range(len(prob[i]))])
    Rad_Score['Score'] = np.ravel([prob[i][j] for i in range(len(dataset)) for j in range(len(prob[i]))])
    Rad_Score['Class'] = np.ravel([class_type[i][j] for i in range(len(dataset)) for j in range(len(class_type[i]))])
    plt.figure(figsize=(5,5))
    sns.violinplot(x="Dataset", y="Score", hue="Class", data=Rad_Score, palette="muted", split=True, inner='quartile')
    plt.ylabel('Prediction Score')
    plt.legend(loc="best",edgecolor='k',fontsize=10,fancybox=False)
    plt.title('Fusion Model')
    plt.text(-0.4,0.3,'P=%.1e'%(t_train[1]),bbox=dict(boxstyle="square",
    ec=(1., 0.5, 0.5),  fc=(1., 0.8, 0.8),))
    plt.text(0.6,0.3,'PP=%.1e'%(t_test[1]),bbox=dict(boxstyle="square",
    ec=(1., 0.52, 0.5),  fc=(1., 0.8, 0.8),))
    plt.subplots_adjust(top=0.99,bottom=0.05,left=0.105,right=0.99,hspace=0,wspace=0)
    
  
    
    plt.figure(figsize=(5,5)) 
    plt.plot(fpr_train_Fusion,tpr_train_Fusion, color='r',linestyle='-',
              lw=lw, label='All Feature\nAUC=%.2f'%train_auc_value+u"\u00B1"+'%.2f, 95%% CI:%.2f-%.2f'%(train_auc_fstd, train_auc_fl, train_auc_fh))
    plt.plot(fpr_train_Rad,tpr_train_Rad, color='b',linestyle='-',
              lw=lw, label='CT Radiomics Feature\nAUC=%.2f'%auc_score_train_Rad+u"\u00B1"+'%.2f, 95%% CI:%.2f-%.2f'%(auc_std_train_Rad, auc_l_train_Rad, auc_h_train_Rad))
    plt.plot(fpr_train_Clinical,tpr_train_Clinical, color='g',linestyle='-',
              lw=lw, label='Clinical Feature\nAUC=%.2f'%auc_score_train_Clinical+u"\u00B1"+'%.2f, 95%% CI:%.2f-%.2f'%(auc_std_train_Clinical, auc_l_train_Clinical, auc_h_train_Clinical))
    plt.xlim([-0.01, 1.01])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves')
    plt.legend(loc="lower right",edgecolor='k',title='Training Cohort',fontsize=10,fancybox=False)
    plt.subplots_adjust(top=0.985,bottom=0.095,left=0.115,right=0.975,hspace=0,wspace=0)
    
    plt.figure(figsize=(5,5)) 
    plt.plot(fpr_test_Fusion,tpr_test_Fusion, color='r',linestyle='-',
              lw=lw, label='All Feature\nAUC=%.2f'%test_auc_value+u"\u00B1"+'%.2f, 95%% CI:%.2f-%.2f'%(test_auc_fstd, test_auc_fl, test_auc_fh))
    plt.plot(fpr_Rad,tpr_Rad, color='b',linestyle='-',
              lw=lw, label='CT Radiomics Feature\nAUC=%.2f'%auc_score_Rad+u"\u00B1"+'%.2f, 95%% CI:%.2f-%.2f'%(auc_std_Rad, auc_l_Rad, auc_h_Rad))
    plt.plot(fpr_Clinical,tpr_Clinical, color='g',linestyle='-',
              lw=lw, label='Clinical Feature\nAUC=%.2f'%auc_score_Clinical+u"\u00B1"+'%.2f, 95%% CI:%.2f-%.2f'%(auc_std_Clinical, auc_l_Clinical, auc_h_Clinical))
    plt.xlim([-0.01, 1.01])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('ROC Curves')
    plt.legend(loc="lower right",edgecolor='k',title='Validation Cohort',fontsize=10,fancybox=False)
    plt.subplots_adjust(top=0.985,bottom=0.095,left=0.115,right=0.975,hspace=0,wspace=0)
