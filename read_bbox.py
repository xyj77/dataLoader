#-*- coding:utf-8 -*-
import os
import re
import pandas
import numpy as np
import scipy.io as sio
import scipy.misc as misc

BBOX_DATA_DIR = '../../Data/BboxAug'
MAT_DATA_DIR = '/media/lab304/J52/Dataset/Mat'
LABEL_PATH = '../../Data/labels.csv'
SAVE_DIR = '../../Data/ROI'

def readLabel(asIndex = 'modalNo', index = 'A'):
    '''Read labels
        Read labels as request.

    Args:
        asIndex: String, Index type.
        index: String, Index.

    Returns:
        labels: A DataFrame of reorganized labels.
        For example:

                            serNo                 Location        Center meanSize  \
        tumourNo modalNo                                                          
        1        A            7    [257,167,301,236,5,8]   [279,201,7]  [42,64]   
                 B            8    [255,178,300,237,6,8]   [277,207,8]  [42,64]   
                              ...
                 J           22  [247,170,287,234,14,24]  [267,202,22]  [42,64]   
        2        A            8    [214,182,254,229,6,9]   [234,205,8]  [35,42]   
                 B            8    [218,183,253,227,7,9]   [235,205,8]  [35,42]   
                              ...
                 J           25  [206,186,240,227,23,28]  [223,206,25]  [35,42]   
        3        A            9    [281,142,303,166,8,9]   [292,154,9]  [19,22]   
                              ... 

        Or:
                              serNo                 Location        Center   meanSize  \
        patientNo  tumourNo                                                            
        '00070993' 1             8   [223,176,284,242,6,10]   [253,209,8]    [62,65]   
        '00090960' 1             5    [191,139,224,184,5,5]   [207,161,5]    [31,42]   
        '00159253' 1            13  [231,149,288,206,11,15]  [259,177,13]    [56,57]   
        '00190415' 1             7    [257,167,301,236,5,8]   [279,201,7]    [42,64]   
                   2             8    [214,182,254,229,6,9]   [234,205,8]    [35,42]   
                   3             9    [281,142,303,166,8,9]   [292,154,9]    [19,22]   
        '00431620' 1            15    [245,64,348,172,9,17]  [296,118,15]   [97,103]   
        '00525849' 1             9    [153,216,183,242,9,9]   [168,229,9]    [28,25]   
        '00582685' 1            12  [264,104,317,159,10,13]  [290,131,12]    [56,59] 
                               ...        

    Raises:
        None
    Usage:
        readLabel(asIndex = 'patientNo', index = '00190415')  
        readLabel(asIndex = 'modalNo', index = 'A') 
    '''
    #读csv文件
    if asIndex is 'modalNo':
        labels = pandas.read_csv(LABEL_PATH, index_col=[2,0,1])
    elif asIndex is 'patientNo':
        index = '\'' + index + '\''
        labels = pandas.read_csv(LABEL_PATH, index_col=[0,1,2])
    labels = labels.fillna('null')
    ''' DataFeame usage:
    # print(labels.dtypes)
    # print(labels.iloc[0])
    # print(labels.loc[('\'00190415\'', 2, 'B'), :])
    # print(labels.loc[('\'00190415\'', 2, 'B'), 'WHO'])
    # print(labels.loc['\'00190415\'', 2, 'B']['WHO'])
    '''
    return labels.loc[index]

def readBbox(liverVolume, tumourInfo, saveNeg=False):
    pattern = re.compile(r'[\d]')
    tumourLoc = [int(x) for x in tumourInfo['Location'][1:-1].split(',') if pattern.search(x)]
    tumourCenter = [int(x) for x in tumourInfo['Center'][1:-1].split(',') if pattern.search(x)]
    tumourSize = [int(x) for x in tumourInfo['meanSize'][1:-1].split(',') if pattern.search(x)]
    if saveNeg:
        tumourD = [int(x) for x in tumourInfo['d'][1:-1].split(',') if pattern.search(x)]
        tumourCenter[0], tumourCenter[1] = tumourCenter[0] + tumourD[0], tumourCenter[1] + tumourD[1]
        
    return liverVolume[tumourCenter[0]-tumourSize[0]/2:tumourCenter[0]+tumourSize[0]/2+1,
                       tumourCenter[1]-tumourSize[1]/2:tumourCenter[1]+tumourSize[1]/2+1,
                       tumourLoc[4]:tumourLoc[5]+1]

def saveSlice(patientNo, tumourNo, modal, Bbox, tumourInfo, saveNeg=False):
    pattern = re.compile(r'[\d]')
    tumourLoc = [int(x) for x in tumourInfo['Location'][1:-1].split(',') if pattern.search(x)]
    tumourWHO = tumourInfo['WHO']
    tumourEd = int(tumourInfo['Edmondson'])
    saveDir = os.path.join(os.path.join(SAVE_DIR, modal), 'Pos')
    if saveNeg:
        saveDir = os.path.join(os.path.join(SAVE_DIR, modal), 'Neg')
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)    
    for slice in range(tumourLoc[4], tumourLoc[5]+1):
        sampleName = patientNo + '_' + str(tumourNo) + '_' + modal + '_' + str(slice) + '.jpg'
        roi = Bbox[:, :, slice-tumourLoc[4]]
        savePath = os.path.join(saveDir, sampleName)
        misc.imsave(savePath, roi)
        print(savePath+' saved!')
        
def saveFusion(patientNo, tumourNo, negBboxs, BboxInfo, saveNeg=True):
    pass

def readModalData(modal = 'A'):
    '''
    指定模态读取数据
    '''   
    if modal is 'K':
        labels = readLabel(asIndex = 'modalNo', index = 'B')
    else:
        labels = readLabel(asIndex = 'modalNo', index = modal)
    patientList = os.listdir(MAT_DATA_DIR)
    for patientNo in patientList:
        # 读取MRI体数据 512x512xS
        dataDir = os.path.join(MAT_DATA_DIR, patientNo)
        dataPath = os.path.join(dataDir, modal+'.mat')
        liverVolume = sio.loadmat(dataPath)['D']
        # 读取tumour信息
        patientInfo = labels.loc['\'' + patientNo + '\'']
        for tumourNo in range(len(patientInfo)):
            # print(patientNo, tumourNo)
            # 读取肿瘤信息
            tumourInfo = patientInfo.iloc[tumourNo]
            # 读取Bbox
            posBbox = readBbox(liverVolume, tumourInfo)
            negBbox = readBbox(liverVolume, tumourInfo, saveNeg=True)
            # 保存切片
            saveSlice(patientNo, tumourNo, modal, posBbox, tumourInfo)
            saveSlice(patientNo, tumourNo, modal, negBbox, tumourInfo, saveNeg=True)
            

def readPatientData(Fusion = ['A', 'B', 'K']):
    '''
    按照病人读取多个模态数据
    '''
    patientList = os.listdir(MAT_DATA_DIR)
    for patientNo in patientList:
        # 读取病人的标注信息
        labels = readLabel(asIndex = 'patientNo', index = patientNo)
        for tumourNo in range(len(labels)/8):
            Info = labels.loc[tumourNo+1]
            posBboxs, negBboxs, BboxInfo = [], [], []
            for modal in Fusion:
                # 读取MRI体数据 512x512xS
                dataDir = os.path.join(MAT_DATA_DIR, patientNo)
                dataPath = os.path.join(dataDir, modal+'.mat')
                liverVolume = sio.loadmat(dataPath)['D']
                if modal is 'K':
                    tumourInfo = Info.loc['B']
                else:
                    tumourInfo = Info.loc[modal]            
                # 读取Bbox
                posBbox = readBbox(liverVolume, tumourInfo)
                negBbox = readBbox(liverVolume, tumourInfo, saveNeg=True)            
                posBboxs.append(posBbox)
                negBboxs.append(negBbox)
                BboxInfo.append(tumourInfo)           
            
            print(len(posBboxs), len(negBboxs), len(BboxInfo))
            saveFusion(patientNo, tumourNo, posBboxs, BboxInfo)
            saveFusion(patientNo, tumourNo, negBboxs, BboxInfo, saveNeg=True)
            
        raw_input()
        
        
def main():
    # # 按照模态读取
    # readModalData(modal='A')
    # readModalData(modal='B')
    # readModalData(modal='K')
    # 按照个体读取
    readPatientData(Fusion = ['A', 'B', 'K'])
  
if __name__ == "__main__":
    main()