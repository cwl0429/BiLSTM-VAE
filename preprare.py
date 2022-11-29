import os
import pickle
from utils import joint
import numpy as np
from tqdm import tqdm
from processing import calculate_angle, calculate_position
import random
import shutil
from pykalman import KalmanFilter

def Kalman1D(observations,damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, _ = kf.smooth(observations)
    return pred_state

def kalmanFilter(data):
    kalman = [Kalman1D(joint, 0.05) for joint in data.T]
    kalman = np.array(kalman).T[0]
    return kalman


def normalize(data):
    data = data.reshape(data.shape[0], int(data.shape[1]/3), 3)
    normal_data = []
    for i, frame in enumerate(data):
        root = (frame[joint['rthigh']]+frame[joint['lthigh']])/2
        data[i, joint['pelvis']] = root
        normal_data.append([])
        for node in frame:
            normal_data[-1].extend(node - root)
    return np.array(normal_data)

def arugment(data):
    arug_data = []
    arug_data.extend(data)
    while len(arug_data) < 100:
        slow = 3 # 內插frame
        for i in range(slow):
            arug_data.append(data[-1]+i*(data[0]-data[-1])/(slow+1))
        arug_data.extend(data)
    return np.array(arug_data)


def basic():
    srcPath = '../Dataset/ChoreoMaster'
    for act in os.listdir(srcPath):
        outPath = f'../Dataset/ChoreoMaster_Normal/{act}'
        files = os.listdir(f'{srcPath}/{act}')
        if not os.path.isdir(outPath):
            os.mkdir(outPath)
        for i, file in enumerate(tqdm(files)):
            filePath = f'{srcPath}/{act}/{file}'
            savePath = f'{outPath}/{act[0]}_act_{i+1}_ca_01.pkl'
            with open(filePath, 'rb') as fpick:
                data = pickle.load(fpick)
            normal_data = normalize(data)  

            if len(normal_data) < 100:
                print("arugment!")
                normal_data = arugment(normal_data)
            
            with open(savePath, 'wb') as fpick:
                pickle.dump(normal_data, fpick)

def divide():# 分成train test
    srcPath = '../Dataset/ChoreoMaster_Normal'
    subjects = os.listdir(srcPath)
    testPath = '../Dataset/ChoreoMaster_Normal/test'
    trainPath = '../Dataset/ChoreoMaster_Normal/train'
    if not os.path.isdir(testPath):
        os.mkdir(testPath)
    if not os.path.isdir(trainPath):
        os.mkdir(trainPath)

    for sub in subjects:
        print(sub)
        Path = f'{srcPath}/{sub}'
        files = os.listdir(Path)
        testfiles = random.sample(files, int(len(files)*0.2)) #test比例
        for file in tqdm(files):
            filePath = f'{Path}/{file}'
            if file in testfiles:
                shutil.copyfile(filePath, f'{testPath}/{file}')
            else:
                shutil.copyfile(filePath, f'{trainPath}/{file}')

def convertAngle():
    srcPath = '../Dataset/ChoreoMaster_Normal'
    folders = os.listdir(srcPath)
    for folder in folders:
        Path = f'{srcPath}/{folder}'
        outPath = f'{Path}_angle'
        files = os.listdir(Path)
        if not os.path.isdir(outPath):
            os.mkdir(outPath)
        for file in tqdm(files):
            filePath = f'{Path}/{file}'
            savePath = f'{outPath}/{file}'
            with open(filePath, 'rb') as fpick:
                data = pickle.load(fpick)
            angle_data = calculate_angle(data)  
            
            with open(savePath, 'wb') as fpick:
                pickle.dump(angle_data, fpick)

def prepare_cvae_data(path, index, output):
    with open(path, 'rb')as fpick:
        data = pickle.load(fpick)
    data = data[index]
    with open(f'../Dataset/CVAE/demo/{output}', 'wb')as fpick:
        angle_data = calculate_angle(data)
        pickle.dump(angle_data, fpick)

def add_noise(path, output):
    with open("../Dataset/Human3.6M/train/s_01_act_02_subact_01_ca_01.pickle", 'rb')as fpick:
        TPose = pickle.load(fpick)[0]
    with open(path, 'rb')as fpick:
        data = pickle.load(fpick)
    with open(f'result/gt_{output}', 'wb')as fpick:
        pickle.dump(calculate_position(data, TPose), fpick)
    noise = np.array([0.04 * np.random.normal(0, 1, len(data)) for _ in range(45)])    
    data = data + noise.T
    data_kalman = kalmanFilter(calculate_position(data, TPose))
    with open(f'../Dataset/Human3.6M/demo/{output}', 'wb')as fpick:
        pickle.dump(data, fpick)
    with open(f'result/kalman_{output}', 'wb')as fpick:
        pickle.dump(data_kalman, fpick)

def connect(path_1, path_2):
    with open(path_1, 'rb')as fpick:
        data_1 = pickle.load(fpick)[30:40]
    with open(path_2, 'rb')as fpick:
        data_2 = pickle.load(fpick)[10:20]
    mask = np.ones((30, 45))
    data = np.concatenate((data_1, mask, data_2), 0)
    with open(f'../Dataset/Human3.6M/demo/fw.pkl', 'wb')as fpick:
        pickle.dump(data, fpick)
    return data
    
if __name__=='__main__':
    # basic()
    # divide()
    convertAngle()
    # # path = "result/cvae/Walk_gt_0004_3_60.pkl"
    # path = "../Dataset/Human3.6M/test_angle/s_09_act_02_subact_02_ca_01.pickle"
    # index = 50
    # # output = 'walk_60.pkl'
    # output ="S09_02_02.pkl"
    # add_noise(path, output)
    # # prepare_cvae_data(path, index, output)
    # path_1 = '../Dataset/Human3.6M/train_angle/s_01_act_02_subact_01_ca_01.pickle'
    # path_2 = '../Dataset/Human3.6M/train_angle/s_01_act_03_subact_01_ca_01.pickle'
    # data = connect(path_1, path_2)
    # print(data.shape)