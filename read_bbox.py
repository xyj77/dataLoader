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
SAVE_DIR = '../../Data'

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

def saveSlice(patientNo, tumourNo, modal, Bbox, tumourInfo, standard, saveNeg=False):
    if saveNeg and patientNo in ['00431620', '03930451']:
        return
    pattern = re.compile(r'[\d]')
    tumourLoc = [int(x) for x in tumourInfo['Location'][1:-1].split(',') if pattern.search(x)]
    serNo = tumourInfo['serNo']
    tumourWHO = tumourInfo['WHO']
    tumourEd = int(tumourInfo['Edmondson'])
    # 确定存储目录
    saveDir = os.path.join(os.path.join(SAVE_DIR, standard), modal)
    if standard is 'Binary':
        if saveNeg:
            saveDir = os.path.join(saveDir, '0')
        else:
            saveDir = os.path.join(saveDir, '1')
    else:
        if standard is 'WHO':
            saveDir = os.path.join(saveDir, str(tumourWHO-1))
        else:
            saveDir = os.path.join(saveDir, str(tumourEd-1))
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
        
    up, bottom = serNo-tumourLoc[4], tumourLoc[5]-serNo
    up, bottom = int(up*0.75+0.5), int(bottom*0.75+0.5)
    # 保存中间层面
    roi = Bbox[:, :, up]
    sampleName = patientNo + '_' + str(tumourNo) + '_' + modal + '_0_'\
                           + '_' + str(tumourWHO) + '_' + str(tumourEd) + '.jpg'
    savePath = os.path.join(saveDir, sampleName)
    misc.imsave(savePath, roi)
    
    print(savePath+' saved!')
    
    # 保存上面层面
    for i in range(up):
        sampleName = patientNo + '_' + str(tumourNo) + '_' + modal + '_' + str(-1-i)\
                               + '_' + str(tumourWHO) + '_' + str(tumourEd) + '.jpg'
        roi = Bbox[:, :, up-i-1]
        savePath = os.path.join(saveDir, sampleName)
        misc.imsave(savePath, roi)
        print(savePath+' saved!')
        
    # 保存下面层面
    for i in range(bottom):
        sampleName = patientNo + '_' + str(tumourNo) + '_' + modal + '_' + str(i+1)\
                               + '_' + str(tumourWHO) + '_' + str(tumourEd) + '.jpg'
        roi = Bbox[:, :, up+i+1]
        savePath = os.path.join(saveDir, sampleName)
        misc.imsave(savePath, roi)
        print(savePath+' saved!')
        
def saveFusion(patientNo, tumourNo, Bboxs, BboxInfo, fusionName, standard, saveTpye, saveNeg=False):
    if saveNeg and patientNo in ['00431620', '03930451']:
        return
    tumourWHO = BboxInfo[0]['WHO']
    tumourEd = int(BboxInfo[0]['Edmondson'])
    # 确定存储目录
    saveDir = os.path.join(os.path.join(SAVE_DIR, standard), fusionName)
    if standard is 'Binary':
        if saveNeg:
            saveDir = os.path.join(saveDir, '0')
        else:
            saveDir = os.path.join(saveDir, '1')
    else:
        if standard is 'WHO':
            saveDir = os.path.join(saveDir, str(tumourWHO-1))
        else:
            saveDir = os.path.join(saveDir, str(tumourEd-1))
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    tumourSize = Bboxs[0].shape
    # 保存融合图像
    picMat = np.zeros((tumourSize[0], tumourSize[1], len(BboxInfo)))
    
    # 计算能够在上下两端保存多少张图像
    upSlice, bottomSlice = [], [] #中间序列所在层面(同时对应中间层面上面的层面数), 中间层面下的层面数
    for info in BboxInfo:
        tumourLoc = [int(x) for x in info['Location'][1:-1].split(',')]
        serNo, Z1, Z2 = info['serNo'], tumourLoc[4], tumourLoc[5]
        upSlice.append(serNo-Z1)
        bottomSlice.append(Z2-serNo)
    up, bottom = min(upSlice), min(bottomSlice)
    print(upSlice, bottomSlice, up, bottom, int(up*0.75+0.5), int(bottom*0.75+0.5))
    # 去掉过于边缘的图像
    up, bottom = int(up*0.75+0.5), int(bottom*0.75+0.5)
    # 保存中间层面
    for index, info in enumerate(BboxInfo):
        picMat[:, :, index] = Bboxs[index][:, :, upSlice[index]]
    sampleName = patientNo + '_' + str(tumourNo) + '_' + fusionName + '_0'\
                           + '_' + str(tumourWHO) + '_' + str(tumourEd) + saveTpye
    savePath = os.path.join(saveDir, sampleName)
    if saveTpye is 'jpg':
        misc.imsave(savePath, picMat)
    else:
        np.save(savePath, picMat)
    print(savePath+' saved!')
    
    # 保存上面层面
    for i in range(up):
        for index, info in enumerate(BboxInfo):
            picMat[:, :, index] = Bboxs[index][:, :, upSlice[index]-i-1]
        sampleName = patientNo + '_' + str(tumourNo) + '_' + fusionName + '_' + str(-1-i)\
                               + '_' + str(tumourWHO) + '_' + str(tumourEd) + saveTpye
        savePath = os.path.join(saveDir, sampleName)
        if saveTpye is '.jpg':
            misc.imsave(savePath, picMat)
        else:
            np.save(savePath, picMat)
        print(savePath+' saved!')
        
    # 保存下面层面
    for i in range(bottom):
        for index, info in enumerate(BboxInfo):
            picMat[:, :, index] = Bboxs[index][:, :, upSlice[index]+i+1]
        sampleName = patientNo + '_' + str(tumourNo) + '_' + fusionName + '_' + str(i+1)\
                               + '_' + str(tumourWHO) + '_' + str(tumourEd) + saveTpye
        savePath = os.path.join(saveDir, sampleName)
        if saveTpye is '.jpg':
            misc.imsave(savePath, picMat)
        else:
            np.save(savePath, picMat)
        print(savePath+' saved!')
    
def readModalData(modal = 'A', standard = 'WHO'):
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
            # roi区域
            posBbox = readBbox(liverVolume, tumourInfo)
            saveSlice(patientNo, tumourNo, modal, posBbox, tumourInfo, standard)
            
            if standard is 'Binary':
                # 背景区域
                negBbox = readBbox(liverVolume, tumourInfo, saveNeg=True)
                saveSlice(patientNo, tumourNo, modal, negBbox, tumourInfo, standard, saveNeg=True)
            
def readPatientData(Fusion = ['A', 'B', 'K'], standard = 'WHO', saveTpye = '.jpg'):
    '''
    按照病人读取多个模态数据
    '''
    patientList = os.listdir(MAT_DATA_DIR)
    for patientNo in patientList:
        # 读取病人的标注信息
        labels = readLabel(asIndex = 'patientNo', index = patientNo)
        for tumourNo in range(len(labels)/8):
            Info = labels.loc[tumourNo+1]
            posBboxs, negBboxs, BboxInfo, fusionName = [], [], [], ''
            for modal in Fusion:
                # 读取MRI体数据 512x512xS
                fusionName = fusionName + modal
                dataDir = os.path.join(MAT_DATA_DIR, patientNo)
                dataPath = os.path.join(dataDir, modal+'.mat')
                liverVolume = sio.loadmat(dataPath)['D']
                if modal is 'K':
                    tumourInfo = Info.loc['B']
                else:
                    tumourInfo = Info.loc[modal]            
                # 读取Bbox
                posBbox = readBbox(liverVolume, tumourInfo)
                posBboxs.append(posBbox)
                BboxInfo.append(tumourInfo)
                if standard is 'Binary':
                    negBbox = readBbox(liverVolume, tumourInfo, saveNeg=True) 
                    negBboxs.append(negBbox)                    
            
            saveFusion(patientNo, tumourNo, posBboxs, BboxInfo, fusionName, standard, saveTpye)
            if standard is 'Binary':
                saveFusion(patientNo, tumourNo, negBboxs, BboxInfo, fusionName, standard, saveTpye, saveNeg=True)
        
def main():
    # modalList = ['A', 'B', 'K', 'E', 'F', 'G', 'H', 'I', 'J']
    # fusionList = ['ABK', 'EGJ']
    # # 按照模态读取
    # for modal in modalList:
        # readModalData(modal=modal, standard = 'WHO')
        
    # # 按照个体读取
    # for fusion in fusionList:
        # readPatientData(Fusion = list(fusion), standard = 'WHO')
    
    # # 按照模态读取
    # for modal in modalList:
        # readModalData(modal=modal, standard = 'Edmondson')
        
    # # 按照个体读取
    # for fusion in fusionList:
        # readPatientData(Fusion = list(fusion), standard = 'Edmondson')
        
        # # 按照模态读取
    # for modal in modalList:
        # readModalData(modal=modal, standard = 'Binary')
        
    # # 按照个体读取
    # for fusion in fusionList:
        # readPatientData(Fusion = list(fusion), standard = 'Binary')
        
    # fusionList = ['EFGHIJ', 'ABKEFGHIJ']   
    fusionList = ['ABKEFGHIJ']     
    # 按照个体读取
    for fusion in fusionList:
        readPatientData(Fusion = list(fusion), standard = 'WHO', saveTpye = '.npy')

    for fusion in fusionList:
        readPatientData(Fusion = list(fusion), standard = 'Edmondson', saveTpye = '.npy')

    for fusion in fusionList:
        readPatientData(Fusion = list(fusion), standard = 'Binary', saveTpye = '.npy')
  
if __name__ == "__main__":
    main()