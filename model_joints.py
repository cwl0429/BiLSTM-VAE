import numpy as np
import torch
class JointDef:
    def __init__(self) -> None:
        self.n_joints_torso = 21
        self.n_joints_limb = 18

    def cat_torch(self, part, data):
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
        return part_data

    def cat_numpy(self, part, data):
        if part == 'torso':
            part_data = np.concatenate((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        return part_data

class JointDefPrev:
    def __init__(self) -> None:
        self.n_joints_torso = 9
        self.n_joints_limb = 18
        
    def cat_torch(self, part, data):
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
        return part_data

    def cat_numpy(self, part, data):
        if part == 'torso':
            part_data = np.concatenate((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        return part_data