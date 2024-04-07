# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:13:00 2021

@author: Administrator
"""


import numpy as np
import pandas as pd

import xlrd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test 
from lifelines import CoxPHFitter
import os
import seaborn as sns
import matplotlib.pyplot as plt


    
if __name__ == '__main__':
    font = {'family' : 'Times New Roman',
 			'weight' : 'normal',
 			'size'   : 12,}
    plt.rc('font', **font)
    train_path = '../Results/Result_TrainDataset.csv'
    train_list = pd.read_csv(train_path)
    train_prob_Rad = np.array(train_list['RadScore'])
    train_DFS = np.array(train_list['DFS'])
    train_DFS_Time = np.array(train_list['DFS_Time'])
    print('Radiomics Model:')
    pred_label_train = train_prob_Rad>0.5
    plt.figure()
    # Plot high-risk subgroup
    ind_high = [i for i in range(len(pred_label_train)) if pred_label_train[i]==1]
    kmf = KaplanMeierFitter()
    kmf.fit(train_DFS_Time[ind_high], event_observed=train_DFS[ind_high], label="High-risk group")#,timeline=[0,20,40,60]
    ax = kmf.plot()
    # Plot low-risk subgroup
    ind_low = [i for i in range(len(pred_label_train)) if pred_label_train[i]==0]
    kmf2 = KaplanMeierFitter()
    kmf2.fit(train_DFS_Time[ind_low], event_observed=train_DFS[ind_low], label="Low-risk group")
    ax = kmf2.plot(ax=ax)
    # Set Y axis to fixed scale
    ax.set_ylim([0.0, 1.0])
    # plt.legend(loc="lower right")
    plt.xlabel('Survival Time (Month)')
    plt.ylabel('Survival Probability')
    plt.title("DFS-Training Cohort")
    # plt.subplots_adjust(top=0.950,bottom=0.095,left=0.090,right=0.995,hspace=0,wspace=0)
    results = logrank_test(train_DFS_Time[ind_high], train_DFS_Time[ind_low], train_DFS[ind_high], train_DFS[ind_low], alpha=.99)
    results.print_summary()
    plt.text(0.1,0.1,'P=%.2f'%(results.p_value),bbox=dict(boxstyle="square",
    ec=(1., 0.5, 0.5),  fc=(1., 0.8, 0.8),))
    add_at_risk_counts(kmf, kmf2, ax=ax, rows_to_show=['At risk', 'Censored'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.950,bottom=0.36,left=0.125,right=0.995,hspace=0,wspace=0)
    print(results.p_value)
    print(results.test_statistic)
    print('-----------------------------------------------')
    
    train_DFS_Data = {}
    train_DFS_Data['Prediction'] = pred_label_train
    train_DFS_Data['DFS_Event'] = train_DFS
    train_DFS_Data['DFS_Time'] = train_DFS_Time
    train_DFS_Data = pd.DataFrame.from_dict(train_DFS_Data)
    #Create the Cox model
    cph_model_PFS = CoxPHFitter()    
    #Train the model on the data set
    cph_model_PFS.fit(train_DFS_Data, 'DFS_Time', 'DFS_Event')     
    #Print the model summary
    cph_model_PFS.print_summary()
    
    test_path = '../Results/Result_TestDataset.csv'
    test_list = pd.read_csv(test_path)
    test_prob_Rad = np.array(test_list['RadScore'])
    test_DFS = np.array(test_list['DFS'])
    test_DFS_Time = np.array(test_list['DFS_Time'])
    pred_label = test_prob_Rad>0.5
    plt.figure()
    # Plot high-risk subgroup
    ind_high = [i for i in range(len(pred_label)) if pred_label[i]==1]
    kmf = KaplanMeierFitter()
    kmf.fit(test_DFS_Time[ind_high], event_observed=test_DFS[ind_high], label="High-risk group")
    ax = kmf.plot()
    # Plot low-risk subgroup
    ind_low = [i for i in range(len(pred_label)) if pred_label[i]==0]
    kmf2 = KaplanMeierFitter()
    kmf2.fit(test_DFS_Time[ind_low], event_observed=test_DFS[ind_low], label="Low-risk group")
    ax = kmf2.plot(ax=ax)
    plt.legend(loc="lower right")
    # Set Y axis to fixed scale
    ax.set_ylim([0.0, 1.0])
    plt.xlabel('Survival Time (Month)')
    plt.ylabel('Survival Probability')
    plt.title("DFS-Validation Cohort")
    # plt.subplots_adjust(top=0.950,bottom=0.095,left=0.090,right=0.995,hspace=0,wspace=0)
    results = logrank_test(test_DFS_Time[ind_high], test_DFS_Time[ind_low], test_DFS[ind_high], test_DFS[ind_low], alpha=.99)
    # results.print_summary()
    plt.text(0.1,0.1,'P=%.1e'%(results.p_value),bbox=dict(boxstyle="square",
    ec=(1., 0.5, 0.5),  fc=(1., 0.8, 0.8),))
    add_at_risk_counts(kmf, kmf2, ax=ax, rows_to_show=['At risk', 'Censored'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.950,bottom=0.36,left=0.125,right=0.995,hspace=0,wspace=0)
    print(results.p_value)
    print(results.test_statistic)
    print('-----------------------------------------------')
    
    test_DFS_Data = {}
    test_DFS_Data['Prediction'] = pred_label
    test_DFS_Data['DFS_Event'] = test_DFS
    test_DFS_Data['DFS_Time'] = test_DFS_Time
    test_DFS_Data = pd.DataFrame.from_dict(test_DFS_Data)
    #Create the Cox model
    cph_model_PFS = CoxPHFitter()    
    #Test the model on the data set
    cph_model_PFS.fit(test_DFS_Data, 'DFS_Time', 'DFS_Event')     
    #Print the model summary
    cph_model_PFS.print_summary()

    #
    SBRT_path = '../Results/Result_SBRTDataset.csv'
    SBRT_list = pd.read_csv(SBRT_path)
    SBRT_prob_Rad = np.array(SBRT_list['RadScore'])
    SBRT_PFS = np.array(SBRT_list['PFS'])
    SBRT_PFS_Time = np.array(SBRT_list['PFS_Time'])
    SBRT_LLRFS = SBRT_list['LLRFS'] 
    SBRT_LLRFS_Time = SBRT_list['LLRFS_Time'] 
    SBRT_OS = SBRT_list['OS'] 
    SBRT_OS_Time = SBRT_list['OS_Time'] 
    SBRT_pred_label = SBRT_prob_Rad>0.5#0.73
    plt.figure()
    # Plot high-risk subgroup
    ind_high = [i for i in range(len(SBRT_pred_label)) if SBRT_pred_label[i]==1]
    kmf = KaplanMeierFitter()
    kmf.fit(SBRT_LLRFS_Time[ind_high], event_observed=SBRT_LLRFS[ind_high], label="High-risk group")
    ax = kmf.plot()
    # Plot low-risk subgroup
    ind_low = [i for i in range(len(SBRT_pred_label)) if SBRT_pred_label[i]==0]
    kmf2 = KaplanMeierFitter()
    kmf2.fit(SBRT_LLRFS_Time[ind_low], event_observed=SBRT_LLRFS[ind_low], label="Low-risk group")
    ax = kmf2.plot(ax=ax)
    # Set Y axis to fixed scale
    ax.set_ylim([0.0, 1.0])
    plt.legend(loc="lower right")
    plt.xlabel('Survival Time (Month)')
    plt.ylabel('Survival Probability')
    plt.title("RRFS-SBRT Cohort")
    # plt.subplots_adjust(top=0.950,bottom=0.095,left=0.090,right=0.995,hspace=0,wspace=0)
    results = logrank_test(SBRT_LLRFS_Time[ind_high], SBRT_LLRFS_Time[ind_low], SBRT_LLRFS[ind_high], SBRT_LLRFS[ind_low], alpha=.99)
    results.print_summary()
    plt.text(0.1,0.1,'P=%.2f'%(results.p_value),bbox=dict(boxstyle="square",
    ec=(1., 0.5, 0.5),  fc=(1., 0.8, 0.8),))
    add_at_risk_counts(kmf, kmf2, ax=ax, rows_to_show=['At risk', 'Censored'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.950,bottom=0.36,left=0.125,right=0.995,hspace=0,wspace=0)
    print(results.p_value)
    print(results.test_statistic)

    SBRT_LLRFS_Data = {}
    SBRT_LLRFS_Data['Prediction'] = SBRT_pred_label
    SBRT_LLRFS_Data['LLRFS_Event'] = SBRT_LLRFS
    SBRT_LLRFS_Data['LLRFS_Time'] = SBRT_LLRFS_Time
    SBRT_LLRFS_Data = pd.DataFrame.from_dict(SBRT_LLRFS_Data)
    #Create the Cox model
    cph_model_LLRFS = CoxPHFitter()    
    #SBRT the model on the data set
    cph_model_LLRFS.fit(SBRT_LLRFS_Data, 'LLRFS_Time', 'LLRFS_Event')     
    #Print the model summary
    cph_model_LLRFS.print_summary()
    print('-----------------------------------------------')
    
    plt.figure()
    # SBRT_pred_label = SBRT_prob_Rad>0.15
    # Plot high-risk subgroup
    ind_high = [i for i in range(len(SBRT_pred_label)) if SBRT_pred_label[i]==1]
    kmf = KaplanMeierFitter()
    kmf.fit(SBRT_PFS_Time[ind_high], event_observed=SBRT_PFS[ind_high], label="High-risk group")
    ax = kmf.plot()
    # Plot low-risk subgroup
    ind_low = [i for i in range(len(SBRT_pred_label)) if SBRT_pred_label[i]==0]
    kmf2 = KaplanMeierFitter()
    kmf2.fit(SBRT_PFS_Time[ind_low], event_observed=SBRT_PFS[ind_low], label="Low-risk group")
    ax = kmf2.plot(ax=ax)
    # Set Y axis to fixed scale
    ax.set_ylim([0.0, 1.0])
    plt.xlabel('Survival Time (Month)')
    plt.ylabel('Survival Probability')
    plt.title("PFS-SBRT Cohort")
    # plt.subplots_adjust(top=0.950,bottom=0.095,left=0.090,right=0.995,hspace=0,wspace=0)
    results = logrank_test(SBRT_PFS_Time[ind_high], SBRT_PFS_Time[ind_low], SBRT_PFS[ind_high], SBRT_PFS[ind_low], alpha=.99)
    results.print_summary()
    plt.text(0.1,0.1,'P=%.2f'%(results.p_value),bbox=dict(boxstyle="square",
    ec=(1., 0.5, 0.5),  fc=(1., 0.8, 0.8),))
    add_at_risk_counts(kmf, kmf2, ax=ax, rows_to_show=['At risk', 'Censored'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.950,bottom=0.36,left=0.125,right=0.995,hspace=0,wspace=0)
    print(results.p_value)
    print(results.test_statistic)

    SBRT_PFS_Data = {}
    SBRT_PFS_Data['Prediction'] = SBRT_pred_label
    SBRT_PFS_Data['DPFS_Event'] = SBRT_PFS
    SBRT_PFS_Data['PFS_Time'] = SBRT_PFS_Time
    SBRT_PFS_Data = pd.DataFrame.from_dict(SBRT_PFS_Data)
    #Create the Cox model
    cph_model_DFS = CoxPHFitter()    
    #SBRT the model on the data set
    cph_model_DFS.fit(SBRT_PFS_Data, 'PFS_Time', 'PFS_Event')     
    #Print the model summary
    cph_model_DFS.print_summary()
    print('-----------------------------------------------')
    
    plt.figure()
    # Plot high-risk subgroup
    # SBRT_pred_label = SBRT_prob_Rad>0.3
    ind_high = [i for i in range(len(SBRT_pred_label)) if SBRT_pred_label[i]==1]
    kmf = KaplanMeierFitter()
    kmf.fit(SBRT_OS_Time[ind_high], event_observed=SBRT_OS[ind_high], label="High-risk group")
    ax = kmf.plot()
    # Plot low-risk subgroup
    ind_low = [i for i in range(len(SBRT_pred_label)) if SBRT_pred_label[i]==0]
    kmf2 = KaplanMeierFitter()
    kmf2.fit(SBRT_OS_Time[ind_low], event_observed=SBRT_OS[ind_low], label="Low-risk group")
    ax = kmf2.plot(ax=ax)
    plt.legend(loc="lower right")
    # Set Y axis to fixed scale
    plt.xlabel('Survival Time (Month)')
    plt.ylabel('Survival Probability')
    ax.set_ylim([0.0, 1.0])
    plt.title("OS-SBRT Cohort")
    # plt.subplots_adjust(top=0.950,bottom=0.095,left=0.090,right=0.995,hspace=0,wspace=0)
    results = logrank_test(SBRT_OS_Time[ind_high], SBRT_OS_Time[ind_low], SBRT_OS[ind_high], SBRT_OS[ind_low], alpha=.99)
    results.print_summary()
    plt.text(0.1,0.1,'P=%.3f'%(results.p_value),bbox=dict(boxstyle="square",
    ec=(1., 0.5, 0.5),  fc=(1., 0.8, 0.8),))
    add_at_risk_counts(kmf, kmf2, ax=ax, rows_to_show=['At risk', 'Censored'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.950,bottom=0.36,left=0.125,right=0.995,hspace=0,wspace=0)
    print(results.p_value)
    print(results.test_statistic)
    
    SBRT_OS_Data = {}
    SBRT_OS_Data['Prediction'] = SBRT_pred_label
    SBRT_OS_Data['OS_Event'] = SBRT_OS
    SBRT_OS_Data['OS_Time'] = SBRT_OS_Time
    SBRT_OS_Data = pd.DataFrame.from_dict(SBRT_OS_Data)
    #Create the Cox model
    cph_model_OS = CoxPHFitter()    
    #SBRT the model on the data set
    cph_model_OS.fit(SBRT_OS_Data, 'OS_Time', 'OS_Event')     
    #Print the model summary
    cph_model_OS.print_summary()
    print('-----------------------------------------------')
    
    
    print('Fusion Model:')
    train_prob_fusion = np.array(train_list['FusionScore'])
    pred_label_train = train_prob_fusion>0.5
    plt.figure()
    # Plot high-risk subgroup
    ind_high = [i for i in range(len(pred_label_train)) if pred_label_train[i]==1]
    kmf = KaplanMeierFitter()
    kmf.fit(train_DFS_Time[ind_high], event_observed=train_DFS[ind_high], label="High-risk group")
    ax = kmf.plot()
    # Plot low-risk subgroup
    ind_low = [i for i in range(len(pred_label_train)) if pred_label_train[i]==0]
    kmf2 = KaplanMeierFitter()
    kmf2.fit(train_DFS_Time[ind_low], event_observed=train_DFS[ind_low], label="Low-risk group")
    ax = kmf2.plot(ax=ax)
    # Set Y axis to fixed scale
    ax.set_ylim([0.0, 1.0])
    plt.xlabel('Survival Time (Month)')
    plt.ylabel('Survival Probability')
    plt.title("DFS-Training Cohort")
    # plt.subplots_adjust(top=0.950,bottom=0.095,left=0.090,right=0.995,hspace=0,wspace=0)
    results = logrank_test(train_DFS_Time[ind_high], train_DFS_Time[ind_low], train_DFS[ind_high], train_DFS[ind_low], alpha=.99)
    results.print_summary()
    plt.text(0.1,0.1,'P=%.1e'%(results.p_value),bbox=dict(boxstyle="square",
    ec=(1., 0.5, 0.5),  fc=(1., 0.8, 0.8),))
    add_at_risk_counts(kmf, kmf2, ax=ax, rows_to_show=['At risk', 'Censored'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.950,bottom=0.36,left=0.125,right=0.995,hspace=0,wspace=0)
    print(results.p_value)
    print(results.test_statistic)
    print('-----------------------------------------------')
    
    train_DFS_Data = {}
    train_DFS_Data['Prediction'] = pred_label_train
    train_DFS_Data['DFS_Event'] = train_DFS
    train_DFS_Data['DFS_Time'] = train_DFS_Time
    train_DFS_Data = pd.DataFrame.from_dict(train_DFS_Data)
    #Create the Cox model
    cph_model_PFS = CoxPHFitter()    
    #Train the model on the data set
    cph_model_PFS.fit(train_DFS_Data, 'DFS_Time', 'DFS_Event')     
    #Print the model summary
    cph_model_PFS.print_summary()
    
    test_prob_fusion = np.array(test_list['FusionScore'])
    pred_label = test_prob_fusion>0.5#0.57
    plt.figure()
    # Plot high-risk subgroup
    ind_high = [i for i in range(len(pred_label)) if pred_label[i]==1]
    kmf = KaplanMeierFitter()
    kmf.fit(test_DFS_Time[ind_high], event_observed=test_DFS[ind_high], label="High-risk group")
    ax = kmf.plot()
    # Plot low-risk subgroup
    ind_low = [i for i in range(len(pred_label)) if pred_label[i]==0]
    kmf2 = KaplanMeierFitter()
    kmf2.fit(test_DFS_Time[ind_low], event_observed=test_DFS[ind_low], label="Low-risk group")
    ax = kmf2.plot(ax=ax)
    plt.legend(loc="lower right")
    # Set Y axis to fixed scale
    ax.set_ylim([0.0, 1.0])
    plt.xlabel('Survival Time (Month)')
    plt.ylabel('Survival Probability')
    plt.title("DFS-Validation Cohort")
    # plt.subplots_adjust(top=0.950,bottom=0.095,left=0.090,right=0.995,hspace=0,wspace=0)
    results = logrank_test(test_DFS_Time[ind_high], test_DFS_Time[ind_low], test_DFS[ind_high], test_DFS[ind_low], alpha=.99)
    # results.print_summary()
    plt.text(0.1,0.1,'P=%.1e'%(results.p_value),bbox=dict(boxstyle="square",
    ec=(1., 0.5, 0.5),  fc=(1., 0.8, 0.8),))
    add_at_risk_counts(kmf, kmf2, ax=ax, rows_to_show=['At risk', 'Censored'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.950,bottom=0.36,left=0.125,right=0.995,hspace=0,wspace=0)
    print(results.p_value)
    print(results.test_statistic)
    print('-----------------------------------------------')
    
    test_DFS_Data = {}
    test_DFS_Data['Prediction'] = pred_label
    test_DFS_Data['DFS_Event'] = test_DFS
    test_DFS_Data['DFS_Time'] = test_DFS_Time
    test_DFS_Data = pd.DataFrame.from_dict(test_DFS_Data)
    #Create the Cox model
    cph_model_PFS = CoxPHFitter()    
    #Test the model on the data set
    cph_model_PFS.fit(test_DFS_Data, 'DFS_Time', 'DFS_Event')     
    #Print the model summary
    cph_model_PFS.print_summary()

    SBRT_prob_fusion = np.array(SBRT_list['FusionScore'])
    SBRT_pred_label = SBRT_prob_fusion>0.5#0.6 
    
    plt.figure()
    # Plot high-risk subgroup
    ind_high = [i for i in range(len(SBRT_pred_label)) if SBRT_pred_label[i]==1]
    kmf = KaplanMeierFitter()
    kmf.fit(SBRT_LLRFS_Time[ind_high], event_observed=SBRT_LLRFS[ind_high], label="High-risk group")
    ax = kmf.plot()
    # Plot low-risk subgroup
    ind_low = [i for i in range(len(SBRT_pred_label)) if SBRT_pred_label[i]==0]
    kmf2 = KaplanMeierFitter()
    kmf2.fit(SBRT_LLRFS_Time[ind_low], event_observed=SBRT_LLRFS[ind_low], label="Low-risk group")
    ax = kmf2.plot(ax=ax)
    plt.legend(loc="lower right")
    # Set Y axis to fixed scale
    ax.set_ylim([0.0, 1.0])
    plt.xlabel('Survival Time (Month)')
    plt.ylabel('Survival Probability')
    plt.title("RRFS-SBRT Cohort")
    # plt.subplots_adjust(top=0.950,bottom=0.095,left=0.090,right=0.995,hspace=0,wspace=0)
    results = logrank_test(SBRT_LLRFS_Time[ind_high], SBRT_LLRFS_Time[ind_low], SBRT_LLRFS[ind_high], SBRT_LLRFS[ind_low], alpha=.99)
    results.print_summary()
    plt.text(0.1,0.1,'P=%.2f'%(results.p_value),bbox=dict(boxstyle="square",
    ec=(1., 0.5, 0.5),  fc=(1., 0.8, 0.8),))
    add_at_risk_counts(kmf, kmf2, ax=ax, rows_to_show=['At risk', 'Censored'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.950,bottom=0.36,left=0.125,right=0.995,hspace=0,wspace=0)
    print(results.p_value)
    print(results.test_statistic)

    SBRT_LLRFS_Data = {}
    SBRT_LLRFS_Data['Prediction'] = SBRT_pred_label
    SBRT_LLRFS_Data['LLRFS_Event'] = SBRT_LLRFS
    SBRT_LLRFS_Data['LLRFS_Time'] = SBRT_LLRFS_Time
    SBRT_LLRFS_Data = pd.DataFrame.from_dict(SBRT_LLRFS_Data)
    #Create the Cox model
    cph_model_LLRFS = CoxPHFitter()    
    #SBRT the model on the data set
    cph_model_LLRFS.fit(SBRT_LLRFS_Data, 'LLRFS_Time', 'LLRFS_Event')     
    #Print the model summary
    cph_model_LLRFS.print_summary()
    print('-----------------------------------------------')
    
    plt.figure()
    # SBRT_pred_label = SBRT_prob_fusion>0.45
    # Plot high-risk subgroup
    ind_high = [i for i in range(len(SBRT_pred_label)) if SBRT_pred_label[i]==1]
    kmf = KaplanMeierFitter()
    kmf.fit(SBRT_PFS_Time[ind_high], event_observed=SBRT_PFS[ind_high], label="High-risk group")
    ax = kmf.plot()
    # Plot low-risk subgroup
    ind_low = [i for i in range(len(SBRT_pred_label)) if SBRT_pred_label[i]==0]
    kmf2 = KaplanMeierFitter()
    kmf2.fit(SBRT_PFS_Time[ind_low], event_observed=SBRT_PFS[ind_low], label="Low-risk group")
    ax = kmf2.plot(ax=ax)
    # Set Y axis to fixed scale
    ax.set_ylim([0.0, 1.0])
    plt.xlabel('Survival Time (Month)')
    plt.ylabel('Survival Probability')
    plt.title("PFS-SBRT Cohort")
    # plt.subplots_adjust(top=0.950,bottom=0.095,left=0.090,right=0.995,hspace=0,wspace=0)
    results = logrank_test(SBRT_PFS_Time[ind_high], SBRT_PFS_Time[ind_low], SBRT_PFS[ind_high], SBRT_PFS[ind_low], alpha=.99)
    results.print_summary()
    plt.text(0.1,0.1,'P=%.2f'%(results.p_value),bbox=dict(boxstyle="square",
    ec=(1., 0.5, 0.5),  fc=(1., 0.8, 0.8),))
    add_at_risk_counts(kmf, kmf2, ax=ax, rows_to_show=['At risk', 'Censored'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.950,bottom=0.36,left=0.125,right=0.995,hspace=0,wspace=0)
    print(results.p_value)
    print(results.test_statistic)

    SBRT_PFS_Data = {}
    SBRT_PFS_Data['Prediction'] = SBRT_pred_label
    SBRT_PFS_Data['PFS_Event'] = SBRT_PFS
    SBRT_PFS_Data['PFS_Time'] = SBRT_PFS_Time
    SBRT_PFS_Data = pd.DataFrame.from_dict(SBRT_PFS_Data)
    #Create the Cox model
    cph_model_DFS = CoxPHFitter()    
    #SBRT the model on the data set
    cph_model_DFS.fit(SBRT_PFS_Data, 'PFS_Time', 'PFS_Event')     
    #Print the model summary
    cph_model_DFS.print_summary()
    print('-----------------------------------------------')
    
    # SBRT_pred_label = SBRT_prob_fusion<0.41
    plt.figure()
    # Plot high-risk subgroup
    ind_high = [i for i in range(len(SBRT_pred_label)) if SBRT_pred_label[i]==1]
    kmf = KaplanMeierFitter()
    kmf.fit(SBRT_OS_Time[ind_high], event_observed=SBRT_OS[ind_high], label="High-risk group")
    ax = kmf.plot()
    # Plot low-risk subgroup
    ind_low = [i for i in range(len(SBRT_pred_label)) if SBRT_pred_label[i]==0]
    kmf2 = KaplanMeierFitter()
    kmf2.fit(SBRT_OS_Time[ind_low], event_observed=SBRT_OS[ind_low], label="Low-risk group")
    ax = kmf2.plot(ax=ax)
    plt.legend(loc="lower right")
    # Set Y axis to fixed scale
    plt.xlabel('Survival Time (Month)')
    plt.ylabel('Survival Probability')
    ax.set_ylim([0.0, 1.0])
    plt.title("OS-SBRT Cohort")
    # plt.subplots_adjust(top=0.950,bottom=0.095,left=0.090,right=0.995,hspace=0,wspace=0)
    results = logrank_test(SBRT_OS_Time[ind_high], SBRT_OS_Time[ind_low], SBRT_OS[ind_high], SBRT_OS[ind_low], alpha=.99)
    results.print_summary()
    plt.text(0.1,0.1,'P=%.3f'%(results.p_value),bbox=dict(boxstyle="square",
    ec=(1., 0.5, 0.5),  fc=(1., 0.8, 0.8),))
    add_at_risk_counts(kmf, kmf2, ax=ax, rows_to_show=['At risk', 'Censored'])
    plt.tight_layout()
    plt.subplots_adjust(top=0.950,bottom=0.36,left=0.125,right=0.995,hspace=0,wspace=0)
    print(results.p_value)
    print(results.test_statistic)
    
    SBRT_OS_Data = {}
    SBRT_OS_Data['Prediction'] = SBRT_pred_label
    SBRT_OS_Data['OS_Event'] = SBRT_OS
    SBRT_OS_Data['OS_Time'] = SBRT_OS_Time
    SBRT_OS_Data = pd.DataFrame.from_dict(SBRT_OS_Data)
    #Create the Cox model
    cph_model_OS = CoxPHFitter()    
    #SBRT the model on the data set
    cph_model_OS.fit(SBRT_OS_Data, 'OS_Time', 'OS_Event')     
    #Print the model summary
    cph_model_OS.print_summary()
    print('-----------------------------------------------')
  
  