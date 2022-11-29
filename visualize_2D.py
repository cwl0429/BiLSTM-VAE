import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
from utils import joint, jointChain

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, nargs='+', required=True)
parser.add_argument('-t', '--title', type=str)
parser.add_argument('-n', '--frame_num', type=int, default = 30)
parser.add_argument('-m', '--mask', type=int, nargs='+')
parser.add_argument('-o', '--output', type=str, default='result/figure.png')
parser.add_argument('--stride', type=int, default = 5, help='output frames each time')
parser.add_argument('--scale', type=float, default =.5)
parser.add_argument('--mask_sequence', type=int, default=0)
args = parser.parse_args()

def loadData():
    motionList = []
    for i, f in enumerate(args.file):
        with open(f, 'rb') as fpick:
            motion = pickle.load(fpick)
        if args.mask and i == args.mask_sequence:
            start = args.mask[0]
            end = args.mask[1]
            motion[start:end] = 1
        motion = motion[:args.frame_num]
        motion = motion*args.scale
        motion = motion.reshape(motion.shape[0], int(motion.shape[1]/3), 3)
        motion = motion[::args.stride, :, :]
        motionList.append(motion)
    return motionList

def plot_2D(motionList):
    fig, ax = plt.subplots(len(args.file), len(motionList[0]), figsize=(80, 10*len(args.file)))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    if args.title:
        plt.suptitle(args.title, fontsize=50)

    for axi in ax.ravel():
        axi.axis('off')
        axi.axis([-.4, .4, -.4, .4])
        axi.set_aspect('equal')

    motionList = np.reshape(motionList, (len(motionList)*len(motionList[0]), 
                                            motionList[0].shape[1], 3))
                                            
    for axi, frame in zip(ax.ravel(), motionList):
        for chain in jointChain:
            pre_node = joint[chain[0]]
            next_node = joint[chain[1]]
            x = np.array([frame[pre_node, 0], frame[next_node, 0]])
            y = np.array([frame[pre_node, 1], frame[next_node, 1]])
            if chain in jointChain[-6:]:
                # right
                axi.plot(x, y, color="#3498db", linewidth=10)
            else:
                #left
                axi.plot(x, y, color="#e74c3c", linewidth=10)

    plt.savefig(args.output)

motionList = loadData()
plot_2D(motionList)