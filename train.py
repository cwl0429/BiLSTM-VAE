import random
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from tqdm import tqdm
import os, re
import numpy as np
import pickle , json
import loss, vae, processing, utils

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset  = 'ChoreoMaster_Normal'
train_dir  = 'train_angle'
train_ca = '01'
test_dir = 'test_angle'
test_ca = '01'
inp_len = 20
out_len = 60
batch_size = 128
save_path = os.path.join("ckpt", f"1011_{dataset}_{train_dir}_{train_ca}_{inp_len}{out_len}")
epochs = 250
lr = 0.0001
best_loss = 1000

def total_loss(x, output, y, out_len, mean, log_var):
    return loss.motion_loss(x, output, y, out_len, weight_scale=1), loss.velocity_loss(x, output, y, out_len = out_len , weight_scale=5), loss.KL_loss(mean, log_var) 

def save_result(save_path, result):
    with open(save_path+ "/result.txt" , "a") as f:
        f.write(result)
        f.write("\n")

#10 30 10 #input
#50  #output >> 32 #output 

def load_data():
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    # Get data
    train_data = processing.get_data(dataset, train_dir, ca=train_ca,
                            inp_len=inp_len, out_len=out_len, randomInput=False)   
    test_data = processing.get_data(dataset, test_dir, ca=test_ca,
                            inp_len=inp_len, out_len=out_len, randomInput=False)

    dataset_size = len(train_data['x'])
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_ = DataLoader(dataset=TensorDataset(train_data["x"], train_data["y"]),
                        batch_size=batch_size, sampler = train_sampler)
    test_ = DataLoader(dataset=TensorDataset(train_data["x"], train_data["y"]),
                        batch_size=batch_size, sampler = test_sampler)

    return train_, test_, train_sampler, test_sampler

def divide_data(train_sampler, test_sampler, part):
    train_data = processing.get_part_data(dataset, train_dir, ca=train_ca, part=part,
                            inp_len=inp_len, out_len=out_len, randomInput=False)
    train_ = DataLoader(dataset=TensorDataset(train_data["x"], train_data["y"]),
                        batch_size=batch_size, sampler = train_sampler)
    test_ = DataLoader(dataset=TensorDataset(train_data["x"], train_data["y"]),
                        batch_size=batch_size, sampler = test_sampler)
    return train_, test_
    

def load_model(part):
    if part == 'torso':
        dim = 21
    elif part == 'entire':
        dim = 45
    else:
        dim = 18
    # Get model
    E_L = vae.Encoder_LSTM(inp=dim)
    D_L = vae.Decoder_LSTM(inp=dim)

    E_l = vae.Encoder_latent(inp = 512)
    D_l = vae.Decoder_latent()            
    model = vae.MTGVAE(E_L, D_L, E_l, D_l).to(DEVICE)
    model = model.to(DEVICE)

    return model

def train(model, part, train_, test_):
    best_loss = 1000
    optimizer = Adam(model.parameters(), lr=lr)
    losses = utils.AverageMeter()
    lossList = []
    lossdetail = []
    model_path = f'{save_path}/{part}'
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # Training
    for epoch in range(epochs):
        # train
        model.train()
        for i, (x, y) in enumerate(tqdm(train_)): 
            lossDict = {'angle':utils.AverageMeter(), 'vel':utils.AverageMeter(), 'kl':utils.AverageMeter()}
            x_np = x.numpy()
            x = x.to(DEVICE)
            optimizer.zero_grad()
            out, mean, log_var = model(x, inp_len+out_len, inp_len+out_len)
            loss_angle, loss_vel, loss_kl = total_loss(x_np, out, y.to(DEVICE), inp_len+out_len, mean, log_var)
            loss_ = loss_angle + loss_vel +loss_kl
            
            lossDict['angle'].update(loss_angle.item(), x.size(0))
            lossDict['vel'].update(loss_vel.item(), x.size(0))
            lossDict['kl'].update(loss_kl.item(), x.size(0))
            
            losses.update(loss_.item(), x.size(0))
            loss_.backward()
            optimizer.step()

        train_loss = losses.avg
        epoch_loss = (lossDict['angle'].avg, lossDict['vel'].avg, lossDict['kl'].avg)
        losses.reset()

        # eval
        model.eval()
        for i, (x, y) in enumerate(tqdm(test_)):
            lossDict = {'angle':utils.AverageMeter(), 'vel':utils.AverageMeter(), 'kl':utils.AverageMeter()}
            x_np = x.numpy()
            x = x.to(DEVICE)
            out, mean, log_var = model(x, inp_len+out_len, inp_len+out_len)
            
            loss_angle, loss_vel, loss_kl = total_loss(x_np, out, y.to(DEVICE), inp_len+out_len, mean, log_var)
            loss_ = loss_angle + loss_vel +loss_kl
            
            lossDict['angle'].update(loss_angle.item(), x.size(0))
            lossDict['vel'].update(loss_vel.item(), x.size(0))
            lossDict['kl'].update(loss_kl.item(), x.size(0))
            
            losses.update(loss_.item(), x.size(0))

        # save
        if losses.avg < best_loss:
            torch.save(model, model_path + "/best.pth")
            best_loss = losses.avg
        else:
            torch.save(model, model_path + "/last.pth")
        result = "Part = {}, Epoch = {:3}/{}, train_loss = {:10}, test_loss = {:10}".format(part, epoch+1, epochs, round(train_loss, 7), round(losses.avg, 7))
        lossList.append([round(train_loss, 7), round(losses.avg, 7)])
        lossdetail.append([round(epoch_loss[0],7), round(epoch_loss[1],7), round(epoch_loss[2],7), round(lossDict['angle'].avg, 7), round(lossDict['vel'].avg, 7), round(lossDict['vel'].avg, 7)])
        print('Using device:', DEVICE)
        print(result)
        save_result(model_path, result)
        losses.reset()
    with open(model_path + '/loss.pkl', 'wb')as fpick:
        pickle.dump(lossList, fpick)   
    with open(model_path + '/lossdetail.pkl', 'wb')as fpick:
        pickle.dump(lossdetail, fpick)        
    print("done!")

def sample_test():
    path = f'../Dataset/{dataset}/{test_dir}'
    ca = test_ca.split('_')
    testFiles = [file for file in os.listdir(path) if re.split("_|\.", file)[-2] in ca]
    testFiles = random.sample(testFiles, 10)
    for file in testFiles:
        data = processing.get_single_data(dataset, test_dir, file)


if __name__=='__main__':
    # sample_test()
    # save log
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        opt = {"dataset":dataset, "train_dir":train_dir, "train_ca":train_ca, "test_ca":test_ca, 
                "lr":lr, "batch_size":batch_size, "epochs":epochs, "inp_len":inp_len, "out_len":out_len}
        with open(save_path + '/opt.json', 'w') as fp:
            json.dump(opt, fp)
    
    # train fullbody
    # part = 'entire'
    # model = load_model(part)
    # train_, test_, train_sampler, test_sampler = load_data()
    # train(model, part, train_, test_)

    # train part
    partList = ['torso','leftarm', 'rightarm', 'leftleg', 'rightleg']
    train_, test_, train_sampler, test_sampler = load_data()
    for part in partList:
        model = load_model(part)
        train_, test_ = divide_data(train_sampler, test_sampler, part)
        train(model, part, train_, test_)  
