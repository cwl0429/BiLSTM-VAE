import pickle
import numpy as np
import argparse    
import os
from processing import get_single_data, calculate_position
import torch
from visualize import AnimePlot

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# python3 inference.py -t infilling -m seiha_Human3.6M_train_angle_01_1_1010 -d ChoreoMaster_Normal/test_angle -f d_act_1_ca_01.pkl -o result/1011_demo_0.pkl -v -p
# python3 inference.py -t infilling -m 1011_ChoreoMaster_Normal_train_angle_01_2010 -d Tool/ -f NE6101076_1027_160_150_frames_angle.pkl -o result/tool_demo_1027.pkl -v -p
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", type=str, help="Evaluation type", required=True) # infilling/inter/smooth
parser.add_argument("-m", "--model", type=str, help="Model name", required=True) # e.g. Human3.6M_train_angle_01_1_1010
parser.add_argument("-d", "--dataset", type=str, help="Dataset Dir", required=True) # e.g. Human3.6M/test_angle
parser.add_argument("-f", "--file", type=str, help="File name")                          
parser.add_argument("-o", "--out", type=str, help="Output Dir", default='result/demo.pkl')
parser.add_argument("-v", "--visual", help="Visualize", action="store_true")
parser.add_argument("-p","--partial", help = "Partial model", action="store_true")
parser.add_argument("-s","--save", help="save or not", action="store_true")
args = parser.parse_args()

partList = ['leftarm', 'rightarm', 'leftleg', 'rightleg', 'torso']
dataset = args.dataset.split('/')[0]
dir = args.dataset.split('/')[1]
file = args.file
inp_len = int(args.model.split('_')[-1][:2])
out_len = int(args.model.split('_')[-1][-2:])
batch_size = 128
class Inference:
    def __init__(self) -> None:
        with open("../Dataset/Human3.6M/train/s_01_act_02_subact_01_ca_01.pickle", 'rb')as fpick:
            self.TPose = pickle.load(fpick)[0]

    def addNoise(self, motion):
        noise = np.array([0.04 * np.random.normal(0, 1, len(motion)) for _ in range(45)])    
        return motion + noise.T

    def mpjpe(self, y, out):
        sqerr = (out -  y)**2
        distance = np.zeros((sqerr.shape[0], int(sqerr.shape[1]/3))) # distance shape : batch_size, 15
        for i, k in enumerate(np.arange(0, sqerr.shape[1],3)): 
            distance[:, i] = np.sqrt(np.sum(sqerr[:,k:k+3],axis=1))  # batch_size
        return np.mean(distance)

    def load_model(self, part):
        path = os.path.join("ckpt", args.model, part)
        model = torch.load(path + "/best.pth",map_location = DEVICE).to(DEVICE)
        model.eval()
        return model
            
    def load_data(self, file):
        print(">>> Data loaded -->", file)
        data = get_single_data(dataset, dir, file)
        data = torch.tensor(data.astype("float32"))
        return data
    '''
    It's a real interpolation, not use ai model
    '''
    def interpolation(self, data):
        data = list(data.numpy())
        ran = int(inp_len/2)
        result = data[:ran]
        for j in range(0, len(data)-(inp_len+out_len), (ran+out_len)):
            last_frame = result[-1]
            for frame in range(out_len):
                result.append(last_frame+frame*(data[j+ran+out_len]-last_frame)/(out_len+1))
            result.extend(data[j+ran+out_len:j+ran+out_len+ran])
        tail = len(data) - len(result)
        result.extend(data[-tail:])
        return np.array(result)

    def demo(self, dim, model, data):
        motion = data.to(DEVICE)
        motion = motion.view((1, -1, dim))
        inp = motion
        out, _,_ = model(inp, 50, 50)
        result = out
        result = result.view((-1,dim))
        return result

    def infilling(self, dim, model, data):
        motion = data.to(DEVICE)
        motion = motion.view((1, -1, dim))
        result = motion[:, :int(inp_len/2), :]
        ran = int(inp_len/2)
        for j in range(0, len(data)-ran, ran):
            missing_data = torch.ones_like(motion[:, 0:out_len, :])
            inp = torch.cat((result[:, -ran:, :], missing_data, motion[:, j+ ran : j + ran * 2, :]), 1)
            out, _,_ = model(inp, inp_len+out_len, inp_len+out_len)
            result = torch.cat((result, out[:, ran:2 * ran + out_len, :]), 1)
            
        tail = len(data) - len(result.view((-1,dim)))
        if tail > 0:
            result = torch.cat((result, motion[:, -tail:, :]), 1)  
        result = result.view((-1,dim))
        return result

    def infilling_same_len(self, dim, model, data):
        motion = data.to(DEVICE)
        motion = motion.view((1, -1, dim))
        result = motion[:, :int(inp_len/2), :]
        ran = int(inp_len/2)
        for j in range(0, len(data)-(ran+out_len)+1, ran+out_len):
            missing_data = torch.ones_like(motion[:, 0:out_len, :])
            inp = torch.cat((result[:, -ran:, :], missing_data, motion[:, j + out_len +ran: j + out_len + ran*2, :]), 1)
            out, _,_ = model(inp, inp_len+out_len, inp_len+out_len)
            result = torch.cat((result, out[:, ran:2 * ran + out_len, :]), 1)
            
        tail = len(data) - len(result.view((-1,dim)))
        if tail > 0:
            result = torch.cat((result, motion[:, -tail:, :]), 1)  
        result = result.view((-1,dim))
        return result


    def smooth(self, dim, model, data):
        test = data.to(DEVICE)
        test = test.view((1,-1,dim))
        ran = int(inp_len/2)
        result = test[:, :ran, :]
        for j in range(0, len(data)-(inp_len+out_len), out_len):
            missing_data = torch.ones_like(test[:, 0:out_len, :])
            inp = torch.cat((result[:, j:j+ran, :], missing_data, test[:, j+ran+out_len:j+inp_len+out_len, :]), 1)
            out, _,_ = model(inp, inp_len+out_len, inp_len+out_len)                 
            result = torch.cat((result, out[:, ran:ran+out_len, :]), 1)  
        tail = len(data) - len(result.view((-1,dim)))
        if tail > 0:
            result = torch.cat((result, test[:, -tail:, :]), 1)
        result = result.view((-1,dim))
        return result

    def getResult(self, data, model, part):
        if part == 'torso':
            dim = 21
        elif part == 'entire':
            dim = 45
        else:
            dim = 18
        if args.type == 'infilling':
            result = self.infilling(dim, model, data)
        elif args.type == 'infilling_same_len':
            result = self.infilling_same_len(dim, model, data)
        elif args.type == 'smooth':
            result = self.smooth(dim, model, data)
        elif args.type == 'inter':
            result = self.interpolation(data)
            return result
        elif args.type == 'demo':
            result = self.demo(dim, model, data)
        else:
            print('No this type!!')
        return result.detach().cpu().numpy()

    def combine(self, partDatas):
        torso = partDatas['torso']
        larm = partDatas['leftarm']
        lleg = partDatas['leftleg']
        rarm = partDatas['rightarm']
        rleg = partDatas['rightleg']
        result = np.concatenate((torso[:, 0:9], rarm[:, -6:], torso[:, 9:12], larm[:, -6:], torso[:, 12:18], rleg[:, -6:], torso[:, 18:21], lleg[:, -6:]), 1)
        return result

    def main(self):
        data = self.load_data(file)
        if args.partial:
            partDatas = {}
            for part in partList:
                model = self.load_model(part)
                if part == 'torso':
                    part_data = torch.cat((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
                elif part == 'leftarm':
                    part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
                elif part =='rightarm':
                    part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
                elif part == 'leftleg':
                    part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
                elif part == 'rightleg':
                    part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
                    
                partDatas[part] = self.getResult(part_data, model, part)
            pred = self.combine(partDatas)
        else:
            part = 'entire'
            model = self.load_model(part)
            pred = self.getResult(data, model, part)
        pred = calculate_position(pred, self.TPose)
        gt = calculate_position(data, self.TPose)
        if args.type == 'infilling':
            result = np.zeros_like(pred)
            ran = int(inp_len/2)
            for j in range(0, len(gt)-ran+1, ran):
                step = int(j / ran)
                result[(ran + out_len) * step: (ran + out_len) * step + ran] = gt[j: j + ran]
            gt = result
        return gt, pred


if __name__ == '__main__':
    inference = Inference()
    gt, pred = inference.main()
    path = args.out.split('.')
    if args.save:
        with open(f'{path[0]}.pkl', 'wb') as fpick:
            pickle.dump(pred, fpick)
        with open(f'{path[0]}_ori.pkl', 'wb') as fpick:
            pickle.dump(gt, fpick)
    if args.visual:
        figure = AnimePlot()
        labels = ['Predicted', 'Ground Truth']
        figure.set_fig(labels, path[0])
        figure.set_data([pred, gt], 300)
        figure.animate()