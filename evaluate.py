from turtle import shape
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os, re
import numpy as np
import pickle
import processing
from math import sqrt, acos, degrees
from scipy.linalg import sqrtm
from numpy.random import random
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from pykalman import KalmanFilter
import random

TPose = np.array([-9.253895e-16,  1.520002e+00,  5.763618e-02, -7.766325e-16,  1.449797e+00,
  8.383295e-02, -1.108790e-01,  1.402220e+00,  9.068209e-02, -3.611223e-01,
  1.386547e+00,  1.012844e-01, -5.678881e-01,  1.374650e+00,  7.629652e-02,
  1.108794e-01,  1.402220e+00,  9.068206e-02,  3.610229e-01,  1.384938e+00,
  1.081927e-01,  5.678883e-01,  1.374649e+00,  7.805140e-02, -2.947416e-16,
  1.159837e+00,  2.532821e-02, -6.527200e-02,  9.983425e-01,  3.593471e-02,
 -8.738497e-02,  6.003636e-01,  4.306928e-02, -1.108280e-01,  8.868676e-02,
  8.982641e-02,  6.527197e-02,  9.983426e-01,  3.593468e-02,  8.738495e-02,
  6.003633e-01,  4.306933e-02,  1.108280e-01,  8.868688e-02,  8.982645e-02])

jointIndex = {'head':0, 'neck':1, 'rightshoulder':2, 'rightarm':3, 'righthand':4,
'leftshoulder':5, 'leftarm':6, 'lefthand':7, 'pelvis':8, 'rightleg':9, 'rightknee':10, 'rightankle':11,
'leftleg':12, 'leftknee':13, 'leftankle':14}

jointChain = {'Torso':['head', 'neck', 'pelvis'], 'Lhand':['leftshoulder', 'leftarm', 'lefthand'],
'Rhand':['rightshoulder', 'rightarm', 'righthand'], 'Lleg':['leftleg', 'leftknee', 'leftankle'],
'Rleg':['rightleg', 'rightknee', 'rightankle']}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mse = nn.MSELoss()

prefix = 'seiha'
train_dataset  = 'Human3.6M' # Mixamo ChoreoMaster_Normal Human3.6M
train_dir  = 'train_angle'
train_ca = '01_1' #01_1
test_dataset = 'Human3.6M'
test_dir = 'test_angle'
test_ca = '01_1'
inp_len = 20
out_len = 60
batch_size = 128
stage = 5
model_path = os.path.join("ckpt", f"{prefix}_{train_dataset}_{train_dir}_{train_ca}_{inp_len}{out_len}")
partList = ['torso', 'leftarm', 'rightarm','leftleg', 'rightleg']

def calculate_distance(data):
    difference = ((data[1:] - data[:-1]))**2
    difference = difference.reshape(difference.shape[0], 15, 3)
    movement = []
    for frame in difference:
        movement.append([joint.sum() for joint in frame])
    movement = np.array(movement)
    distance = {}
    for part, joints in jointChain.items():
        summary = 0
        for joint in joints:
            summary += movement[:, jointIndex[joint]]
        distance[part] = summary/3*1000
    return distance

def kalman_1D(observations,damping=1):
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

def kalman_filter(data):
    kalman = [kalman_1D(joint, 0.05) for joint in data.T]
    kalman = np.array(kalman).T[0]
    return kalman

def mpjpe(y, out):
    sqerr = (out -  y)**2
    distance = np.zeros((sqerr.shape[0], int(sqerr.shape[1]/3))) # distance shape : batch_size, 15
    for i, k in enumerate(np.arange(0, sqerr.shape[1],3)): 
        distance[:, i] = np.sqrt(np.sum(sqerr[:,k:k+3],axis=1))  # batch_size
    return np.mean(distance)

def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def load_model(part):
    path = os.path.join(model_path, part)
    print('Model:', path)
    model = torch.load(path + "/best.pth",map_location='cuda:0').to(DEVICE)
    model.eval()
    print(' >>> Model loaded --> ', part)
    return model

def load_data(part):
    # Get data
    if part == 'entire': 
        test_data = processing.get_data(test_dataset, test_dir, ca=test_ca,
                                inp_len=inp_len, out_len=out_len, randomInput=False)
    else:
        test_data = processing.get_part_data(test_dataset, test_dir, ca=test_ca, part=part,
                            inp_len=inp_len, out_len=out_len, randomInput=False)   
    print(len(test_data))
    test_ = DataLoader(dataset=TensorDataset(test_data["x"], test_data["y"]),
                        batch_size=batch_size, shuffle=False, num_workers = 0) 
    return test_

def eval_inter(mode):
    mpjpeList = []
    if mode == 'total':
        model = load_model('entire')
        test_ = load_data('entire')
        for i, (x, y) in enumerate(tqdm(test_)): 
            x = x.to(DEVICE)
            out, _, _ = model(x, inp_len+out_len, inp_len+out_len)
            out = out[:, int(inp_len/2):-int(inp_len/2), 27:36] # 只取要生成的部分(去掉頭尾的frames)
            out = out.contiguous().view(-1, 45)
            out = out.detach().cpu().numpy()
            out = processing.calculate_position(out, TPose) # 角度轉座標
            y = y[:, int(inp_len/2):-int(inp_len/2), 27:36] # 只取生成的範圍(去掉頭尾的frames)
            y = y.numpy()
            y = y.reshape(-1,45)
            y = processing.calculate_position(y, TPose) # 角度轉座標
            mpjpeList.append(mpjpe(y, out)) # 計算MPJPE
        # print('MPJPE(mean): %.3f mm' % (np.array(mpjpeList).mean()*1000))

    else:
        models = {}
        test_ = load_data('entire')
        for part in partList:
            models[part] = load_model(part)
        for x, y in tqdm(test_):
            x = x.to(DEVICE)

            # 分部位
            torso_inp = torch.cat((x[:, :, 0:9], x[:, :, 15:18], x[:, :, 24:30], x[:, :, 36:39]), axis=2)
            leftarm_inp = torch.cat((x[:, :, 3:9], x[:, :, 15:18], x[:, :, 24:27], x[:, :, 18:24]), axis=2)
            rightarm_inp = torch.cat((x[:, :, 3:9], x[:, :, 15:18], x[:, :, 24:27], x[:, :, 9:15]), axis=2)
            leftleg_inp = torch.cat((x[:, :, 3:6], x[:, :, 24:30], x[:, :, 36:39], x[:, :, 39:45]), axis=2)
            rightleg_inp = torch.cat((x[:, :, 3:6], x[:, :, 24:30], x[:, :, 36:39], x[:, :, 30:36]), axis=2) 

            torso, _, _ = models["torso"](torso_inp, inp_len+out_len, inp_len+out_len)
            rarm, _, _ = models['rightarm'](rightarm_inp, inp_len+out_len, inp_len+out_len)
            larm, _, _ = models['leftarm'](leftarm_inp, inp_len+out_len, inp_len+out_len)
            rleg, _, _ = models['rightleg'](rightleg_inp, inp_len+out_len, inp_len+out_len)
            lleg, _, _ = models['leftleg'](leftleg_inp, inp_len+out_len, inp_len+out_len) 

            #合併
            out = torch.cat((torso[:, :, 0:9], rarm[:, :, -6:], torso[:, :, 9:12], larm[:, :, -6:], torso[:, :, 12:18], rleg[:, :, -6:], torso[:, :, 18:21], lleg[:, :, -6:]), 2) 
            out = out.contiguous().view(-1, 45)
            out = out.detach().cpu().numpy()
            out = processing.calculate_position(out, TPose) # 角度轉座標
            
            y = y.numpy()
            y = y.reshape(-1,45)
            y = processing.calculate_position(y, TPose) # 角度轉座標
            
            mpjpeList.append(mpjpe(y, out)) # 計算MPJPE
    print('MPJPE(mean): %.3f mm' % (np.array(mpjpeList).mean()*1000))

def eval_inter_each_part():
    mpjpeList_torso = []
    mpjpeList_larm = []
    mpjpeList_rarm = []
    mpjpeList_lleg = []
    mpjpeList_rleg = []

    models = {}
    test_ = load_data('entire')
    for part in partList:
        models[part] = load_model(part)
    for x, y in tqdm(test_):
        x = x.to(DEVICE)
        # 分部位
        torso_inp = torch.cat((x[:, :, 0:9], x[:, :, 15:18], x[:, :, 24:30], x[:, :, 36:39]), axis=2)
        leftarm_inp = torch.cat((x[:, :, 3:9], x[:, :, 15:18], x[:, :, 24:27], x[:, :, 18:24]), axis=2)
        rightarm_inp = torch.cat((x[:, :, 3:9], x[:, :, 15:18], x[:, :, 24:27], x[:, :, 9:15]), axis=2)
        leftleg_inp = torch.cat((x[:, :, 3:6], x[:, :, 24:30], x[:, :, 36:39], x[:, :, 39:45]), axis=2)
        rightleg_inp = torch.cat((x[:, :, 3:6], x[:, :, 24:30], x[:, :, 36:39], x[:, :, 30:36]), axis=2) 
        torso, _, _ = models["torso"](torso_inp, inp_len+out_len, inp_len+out_len)
        rarm, _, _ = models['rightarm'](rightarm_inp, inp_len+out_len, inp_len+out_len)
        larm, _, _ = models['leftarm'](leftarm_inp, inp_len+out_len, inp_len+out_len)
        rleg, _, _ = models['rightleg'](rightleg_inp, inp_len+out_len, inp_len+out_len)
        lleg, _, _ = models['leftleg'](leftleg_inp, inp_len+out_len, inp_len+out_len) 
        #合併
        out = torch.cat((torso[:, :, 0:9], rarm[:, :, -6:], torso[:, :, 9:12], larm[:, :, -6:], torso[:, :, 12:18], rleg[:, :, -6:], torso[:, :, 18:21], lleg[:, :, -6:]), 2) 
        out = out.contiguous().view(-1, 45)
        out = out.detach().cpu().numpy()
        out = processing.calculate_position(out, TPose) # 角度轉座標
        out = divide(out)

        y = y.numpy()
        y = y.reshape(-1,45)
        y = processing.calculate_position(y, TPose) # 角度轉座標
        y = divide(y)
        mpjpeList_torso.append(mpjpe(out['torso'], y['torso']))
        mpjpeList_larm.append(mpjpe(out['leftarm'], y['leftarm']))
        mpjpeList_rarm.append(mpjpe(out['rightarm'], y['rightarm'])) 
        mpjpeList_lleg.append(mpjpe(out['leftleg'], y['leftleg'])) 
        mpjpeList_rleg.append(mpjpe(out['rightleg'], y['rightleg'])) 

    print('Torso MPJPE(mean): %.3f mm' % (np.array(mpjpeList_torso).mean()*1000))
    print('Leftarm MPJPE(mean): %.3f mm' % (np.array(mpjpeList_larm).mean()*1000))
    print('Rightarm MPJPE(mean): %.3f mm' % (np.array(mpjpeList_rarm).mean()*1000))
    print('Leftleg MPJPE(mean): %.3f mm' % (np.array(mpjpeList_lleg).mean()*1000))
    print('Rightleg MPJPE(mean): %.3f mm' % (np.array(mpjpeList_rleg).mean()*1000))

'''
divide data into four part: rhand lhand rleg lleg
'''
def divide(data): 
    data_part = {}
    data_part['torso'] = np.concatenate((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
    data_part['leftarm'] = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
    data_part['rightarm'] = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
    data_part['leftleg'] = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
    data_part['rightleg'] = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
    return data_part
    
def combine(partDatas):
    torso = partDatas['torso']
    larm = partDatas['leftarm']
    lleg = partDatas['leftleg']
    rarm = partDatas['rightarm']
    rleg = partDatas['rightleg']
    result = torch.cat((torso[:, 0:9], rarm[:, -6:], torso[:, 9:12], larm[:, -6:], torso[:, 12:18], rleg[:, -6:], torso[:, 18:21], lleg[:, -6:]), 1)
    return result

def smooth(models, data):
    inp_len = 10
    out_len = 10
    partData = {}
    partData['torso'] = torch.cat((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), 1)
    partData['rightarm'] = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), 1)
    partData['leftarm'] = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), 1)
    partData['rightleg'] = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), 1)
    partData['leftleg'] = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), 1) 
    resultData = {}
    for part in partList:
        data = partData[part]
        test = data.to(DEVICE)
        if part == 'torso':
            dim = 21
        else:
            dim = 18
        test = test.view((1,-1,dim))
        ran = int(inp_len/2)
        result = test[:, :ran, :]
        for j in range(0, len(data)-(inp_len+out_len), out_len):
            missing_data = torch.ones_like(test[:, 0:out_len, :])
            inp = torch.cat((result[:, j:j+ran, :], missing_data, test[:, j+ran+out_len:j+inp_len+out_len, :]), 1)
            out, _,_ = models[part](inp, inp_len+out_len, inp_len+out_len)                 
            result = torch.cat((result, out[:, ran:ran+out_len, :]), 1)  
        tail = len(data) - len(result.view((-1,dim)))
        if tail > 0:
            result = torch.cat((result, test[:, -tail:, :]), 1)
        result = result.view((-1,dim))
        # print(result.shape)
        resultData[part] = result
    final = combine(resultData)
    return final.detach().cpu().numpy()
    
def eval_smooth():
    models = {}
    modelPath = 'ckpt/Mixamo_train_angle_01_1_1010'
    for part in partList:
        print(' >>> Model loaded --> ', part)
        path = os.path.join(modelPath, part)
        print('Model:', path)
        model = torch.load(path + "/best.pth").to(DEVICE)
        model.eval()
        models[part] = model
    mpjpeList = []
    filePath = '../Dataset/Mixamo/train_angle'
    files = os.listdir(filePath)
    maeNoise = {}
    maeDenoise = {}
    maeKalman = {}
    ca = test_ca.split("_")
    fileList = [file for file in files if re.split("_|\.", file)[-2] in ca]
    sampleFile = random.sample(fileList, int(len(fileList)*0.5))
    for file in tqdm(sampleFile):
        with open(f'{filePath}/{file}', 'rb') as fpick:
            data = pickle.load(fpick)
            noise = np.array([0.04 * np.random.normal(0, 1, len(data)) for _ in range(45)])    
            data_noise = data + noise.T
            data_noise = torch.tensor(data_noise.astype("float32"))
            data_denoise = smooth(models, data_noise)
            data_denoise = processing.calculate_position(data_denoise, TPose)
            data_noise = data_noise.numpy()
            data_kalman = kalman_filter(processing.calculate_position(data_noise, TPose))
            data_gt = processing.calculate_position(data, TPose)
            data_noise = processing.calculate_position(data_noise, TPose)
            distance_denoise = calculate_distance(data_denoise)
            distance_noise = calculate_distance(data_noise)
            distance_kalman = calculate_distance(data_kalman)
            distance_gt = calculate_distance(data_gt)
            for part in jointChain.keys():
                if part not in maeNoise.keys():
                    maeNoise[part] = []
                    maeDenoise[part] = []
                    maeKalman[part] = []
                maeNoise[part].append(mean_absolute_error(distance_noise[part], distance_gt[part]))
                maeDenoise[part].append(mean_absolute_error(distance_denoise[part], distance_gt[part]))
                maeKalman[part].append(mean_absolute_error(distance_kalman[part], distance_gt[part]))
    for part in jointChain.keys():
        print(part)
        print('MAE(Noise)', (np.mean(maeNoise[part])))
        print('MAE(Ours)', (np.mean(maeDenoise[part])))
        print('MAE(Kalman)', (np.mean(maeKalman[part])))


if __name__ == "__main__":
    # eval_inter('partial')
    # eval_inter('total')
    eval_inter_each_part()
    # eval_smooth()