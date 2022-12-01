import torch
import torch.nn as nn

MSE = nn.MSELoss()
def motion_loss(x, output, y, out_len=30, weight_scale=1):
    return MSE(output, y)*weight_scale


def velocity_loss(x, output, y, out_len=30, weight_scale=1):
    velocity_gt = y[:, 1:, :]-y[:, :-1, :]
    velocity_output = output[:, 1:, :]-output[:, :-1, :]
    
    # index = np.argwhere(x[0, :, 0] == 1)
    
    # output_1 = torch.cat((y[:, index[0, 0]-1:index[0, 0], :], output[:, index[0, 0]:index[-1, 0]+1, :]), 1)
    # output_2 = torch.cat((output[:, index[0, 0]:index[-1, 0]+1, :], y[:, index[-1, 0]+1:index[-1, 0]+2, :]), 1)

    # velocity_gt = y[:, index[0, 0]-1:index[-1, 0]+1, :] - y[:, index[0, 0]:index[-1, 0]+2, :]
    # velocity_output = output_1-output_2
    
    return MSE(velocity_output, velocity_gt)*weight_scale
    
    # random
    # loss = []
    # for i, frame in enumerate(x):
    #     index = np.argwhere(frame[:, 0] == 1)
    #     velocity_gt = y[i, index[0, 0]-1:index[-1, 0], :] - y[i, index[0, 0]:index[-1, 0]+1, :]
    #     velocity_output = output[i, index[0, 0]-1:index[-1, 0], :] - output[i, index[0, 0]:index[-1, 0]+1, :]
    #     loss.append(MSE(velocity_output, velocity_gt)*weight_scale)
    # return torch.mean(torch.tensor(loss))

def KL_loss(mean, log_var):
    return - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())


if __name__=="__main__":
    x = torch.randn(1, 50, 45)
    x[:, 10:-10, :] = 1
    y = torch.randn(1, 50, 45)
    output = torch.randn(1, 50, 45)
    mt_loss = motion_loss(x, output, y, 50, weight_scale=1)
    print(mt_loss)
    vel_loss = velocity_loss(x, output, y, 50, weight_scale=1)
    print(vel_loss)