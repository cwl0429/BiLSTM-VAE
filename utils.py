joint_line = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [8,9], [9,10], [10,11], [8,12], [12,13], [13,14]]

jointIndex = {"head":0, "neck":3, "rshoulder":6, "rarm":9, "rhand":12, 
                "lshoulder":15, "larm":18, "lhand":21, "pelvis":24, 
                "rthigh":27, "rknee":30,"rankle":33,"lthigh":36, "lknee":39, "lankle":42}

joint = {"head":0, "neck":1, "rshoulder":2, "rarm":3, "rhand":4, 
                "lshoulder":5, "larm":6, "lhand":7, "pelvis":8, 
                "rthigh":9, "rknee":10,"rankle":11,"lthigh":12, "lknee":13, "lankle":14}

jointChain = [["neck","pelvis"], ["head","neck"],  ["rshoulder", "neck"], ["rarm", "rshoulder"], ["rhand", "rarm"],
                ["rthigh", "pelvis"], ["rknee", "rthigh"], ["rankle", "rknee"],
                ["lshoulder", "neck"], ["larm", "lshoulder"], ["lhand", "larm"], 
                ["lthigh", "pelvis"], ["lknee", "lthigh"], ["lankle", "lknee"]]

jointConnect = [(jointIndex[joint[0]], jointIndex[joint[1]]) for joint in jointChain]

class AverageMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum /self.count