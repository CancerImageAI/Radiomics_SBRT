# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:56:11 2021

@author: Administrator
"""


import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from tqdm import tqdm



def readDCM_Img(FilePath):
    reader = sitk.ImageSeriesReader()
    dcm_names = reader.GetGDCMSeriesFileNames(FilePath)
    reader.SetFileNames(dcm_names)
    image = reader.Execute()
    return image

if __name__ == '__main__':
    ## Part 1
    Mask_Path = '../Mask'
    CT_Path = '../CT-Scans'
    Mask_Lists = os.listdir(Mask_Path)
    Error_List = []
    # print('Error List:')
    for mask in tqdm(Mask_Lists):
        Patient_ID = mask.split('.')[0]
        try:
            CT_Img = readDCM_Img(os.path.join(CT_Path,Patient_ID))
            ImgSave_Path = '../Dataset/Image'
            sitk.WriteImage(CT_Img,os.path.join(ImgSave_Path,Patient_ID+'.nii.gz'))
            CT_Mask = sitk.ReadImage(os.path.join(Mask_Path,mask))
            MaskSave_Path = '../Dataset/Mask'
            sitk.WriteImage(CT_Mask,os.path.join(MaskSave_Path,Patient_ID+'.nii.gz'))
        except:
            print(Patient_ID)
            Error_List.append(Patient_ID)    
    ## Part 2        
    Img_Path = '../lunngCTROI'
    Img_Lists = os.listdir(Img_Path)
    for Img_ID in tqdm(Img_Lists):
        Patient_Path = os.path.join(Img_Path,Img_ID)
        Files = os.listdir(Patient_Path)
        for File in Files:
            if len(File.split('.nrrd')[0].split('-'))>1:
                CT_Mask = sitk.ReadImage(os.path.join(Patient_Path,File))
                MaskSave_Path = '../Dataset/Mask'
                sitk.WriteImage(CT_Mask,os.path.join(MaskSave_Path,Img_ID+'.nii.gz'))
            else:
                CT_Img = sitk.ReadImage(os.path.join(Patient_Path,File))
                ImgSave_Path = '../Dataset/Image'
                sitk.WriteImage(CT_Img,os.path.join(ImgSave_Path,Img_ID+'.nii.gz'))
                
                                         
    
