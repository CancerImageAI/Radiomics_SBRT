# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:04:29 2022

@author: Jing
"""

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor,imageoperations

import os
from pandas import DataFrame as DF
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import time
from time import sleep
from tqdm import tqdm
import glob
import xlrd

def Extract_Features(image,mask):
    paramsFile = os.path.abspath('Params.yaml')
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    result = extractor.execute(image, mask)
    general_info = {'diagnostics_Configuration_EnabledImageTypes','diagnostics_Configuration_Settings',
                    'diagnostics_Image-interpolated_Maximum','diagnostics_Image-interpolated_Mean',
                    'diagnostics_Image-interpolated_Minimum','diagnostics_Image-interpolated_Size',
                    'diagnostics_Image-interpolated_Spacing','diagnostics_Image-original_Hash',
                    'diagnostics_Image-original_Maximum','diagnostics_Image-original_Mean',
                    'diagnostics_Image-original_Minimum','diagnostics_Image-original_Size',
                    'diagnostics_Image-original_Spacing','diagnostics_Mask-interpolated_BoundingBox',
                    'diagnostics_Mask-interpolated_CenterOfMass','diagnostics_Mask-interpolated_CenterOfMassIndex',
                    'diagnostics_Mask-interpolated_Maximum','diagnostics_Mask-interpolated_Mean',
                    'diagnostics_Mask-interpolated_Minimum','diagnostics_Mask-interpolated_Size',
                    'diagnostics_Mask-interpolated_Spacing','diagnostics_Mask-interpolated_VolumeNum',
                    'diagnostics_Mask-interpolated_VoxelNum','diagnostics_Mask-original_BoundingBox',
                    'diagnostics_Mask-original_CenterOfMass','diagnostics_Mask-original_CenterOfMassIndex',
                    'diagnostics_Mask-original_Hash','diagnostics_Mask-original_Size',
                    'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_VolumeNum',
                    'diagnostics_Mask-original_VoxelNum','diagnostics_Versions_Numpy',
                    'diagnostics_Versions_PyRadiomics','diagnostics_Versions_PyWavelet',
                    'diagnostics_Versions_Python','diagnostics_Versions_SimpleITK',
                    'diagnostics_Image-original_Dimensionality'}
    features = dict((key, value) for key, value in result.items() if key not in general_info)
    feature_info = dict((key, value) for key, value in result.items() if key in general_info)
    return features,feature_info

if __name__ == '__main__':
    data = xlrd.open_workbook('../data_list/TJCH.xls')
    table = data.sheets()[0]
    List_all = table.col_values(0)[1:]
    Class = table.col_values(-1)[1:]
    Img_Path = '../data_list/Image'
    Mask_Path = '../data_list/Mask'
    Patient_IDs = [i.split('.')[0] for i in os.listdir(Img_Path)]
    PatientName = [''.join([i for i in string if not i.isdigit()]) for string in Patient_IDs]
    PatientName = [i.lower() for i in PatientName]
    ind = [PatientName.index(i) for i in List_all]
    start = time.perf_counter()
    Feature = []
    for i in tqdm(range(len(np.array(Patient_IDs)[ind]))):
        Patient_ID = np.array(Patient_IDs)[ind][i]
        Image = sitk.ReadImage(os.path.join(Img_Path,Patient_ID))
        Mask = sitk.ReadImage(os.path.join(Mask_Path,Patient_ID))
        feature, feature_info = Extract_Features(Image, Mask)
        feature['Patient_ID'] = Patient_ID.split('.')[0]
        feature['Class'] = Class[i]
        Feature.append(feature)
    df = DF(Feature).fillna('0')
    df.to_csv('../Results/TJCH_Radiomics_Feature.csv', index = False, sep=',')
    end = time.perf_counter()
    print(end-start)   
