import os
import subprocess
import numpy as np
import pickle

def compare(c):
    return float(c[1])

# python3 inference.py -t infilling -m 1011_ChoreoMaster_Normal_train_angle_01_2010 -d Tool/ -f NE6101076_1027_160_150_frames_angle.pkl -o result/tool_demo_1027.pkl -v -p
src_path = '../Dataset/Human3.6M/test_angle'
print(os.getcwd())
test_motion_path = os.listdir(src_path)
mpjpe_list = []
for motion_name in test_motion_path:
    # motion_path = os.path.join(src_path, motion_name)
    # with open(motion_path, 'rb') as fpick:
    #     motion = pickle.load(fpick)
    result = subprocess.run(["python3", "inference.py", "-t", "infilling_same_len", "-m", "1011_ChoreoMaster_Normal_train_angle_01_2010", "-d", "Human3.6M/test_angle", "-f", motion_name, "-p"], stdout=subprocess.PIPE, text=True)
    r = result.stdout.split('\n')
    mpjpe_list.append([motion_name, r])

print(mpjpe_list)
mpjpe_list.sort(reverse = True, key = compare)
np.savetxt('mpjpe_Human3_6m_result.txt', np.array(mpjpe_list))   # X is an array
print(mpjpe_list)
