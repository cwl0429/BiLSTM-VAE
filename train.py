import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from tqdm import tqdm
import os
import numpy as np
import pickle , json
import loss, vae, processing, utils
# from model_joints import JointDefV3, JointDefV2, JointDefV1
import argparse    

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Train():
    def __init__(self, joint_def, dataset, train_dir, train_ca, inp_len, out_len, hyperparams) -> None:
        self.joint_def = joint_def
        self.dataset = dataset
        self.train_dir = train_dir
        self.train_ca = train_ca
        self.inp_len = inp_len
        self.out_len = out_len
        self.hyperparams = hyperparams
        
    def total_loss(self, x, output, y, out_len, mean, log_var):
        return loss.motion_loss(x, output, y, out_len, weight_scale=1), loss.velocity_loss(x, output, y, 
                                    out_len = out_len , weight_scale=5), loss.KL_loss(mean, log_var) 

    def save_result(self, save_path, result):
        with open(save_path+ "/result.txt" , "a") as f:
            f.write(result)
            f.write("\n")

    #10 30 10 #input
    #50  #output >> 32 #output 
    def load_data(self):
        validation_split = .2
        shuffle_dataset = True
        random_seed= 42
        # Get data
        train_data = processing.get_data(self.dataset, self.train_dir, ca=self.train_ca,
                                inp_len=self.inp_len, out_len=self.out_len, randomInput=False)   

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
                            batch_size=self.hyperparams['batch_size'], sampler = train_sampler)
        test_ = DataLoader(dataset=TensorDataset(train_data["x"], train_data["y"]),
                            batch_size=self.hyperparams['batch_size'], sampler = test_sampler)

        return train_, test_, train_sampler, test_sampler

    def divide_data(self, train_sampler, test_sampler, part):
        train_data = processing.get_part_data(self.dataset, self.train_dir, ca=self.train_ca, part=part,
                                inp_len=self.inp_len, out_len=self.out_len, randomInput=False, joint_def=joint_def)
        train_ = DataLoader(dataset=TensorDataset(train_data["x"], train_data["y"]),
                            batch_size=self.hyperparams['batch_size'], sampler = train_sampler)
        test_ = DataLoader(dataset=TensorDataset(train_data["x"], train_data["y"]),
                            batch_size=self.hyperparams['batch_size'], sampler = test_sampler)
        return train_, test_
        
    def load_model(self, part):
        dim = self.joint_def.n_joints_part[part]
        # Get model
        E_L = vae.Encoder_LSTM(inp=dim)
        D_L = vae.Decoder_LSTM(inp=dim)

        E_l = vae.Encoder_latent(inp = 512)
        D_l = vae.Decoder_latent()            
        model = vae.MTGVAE(E_L, D_L, E_l, D_l).to(DEVICE)
        model = model.to(DEVICE)

        return model

    def train(self, model, part, train_, test_, save_path):
        best_loss = 1000
        optimizer = Adam(model.parameters(), lr=self.hyperparams['lr'])
        losses = utils.AverageMeter()
        loss_list = []
        loss_detail = []
        model_path = f'{save_path}/{part}'
        if not os.path.isdir(model_path):
            os.mkdir(model_path)

        # Training
        for epoch in range(self.hyperparams['epochs']):
            # train
            model.train()
            for i, (x, y) in enumerate(tqdm(train_)): 
                loss_dict = {'angle':utils.AverageMeter(), 'vel':utils.AverageMeter(), 'kl':utils.AverageMeter()}
                x_np = x.numpy()
                x = x.to(DEVICE)
                optimizer.zero_grad()
                out, mean, log_var = model(x, self.inp_len+self.out_len, self.inp_len+self.out_len)
                loss_angle, loss_vel, loss_kl = self.total_loss(x_np, out, y.to(DEVICE), self.inp_len+self.out_len, mean, log_var)
                loss_ = loss_angle + loss_vel +loss_kl
                
                loss_dict['angle'].update(loss_angle.item(), x.size(0))
                loss_dict['vel'].update(loss_vel.item(), x.size(0))
                loss_dict['kl'].update(loss_kl.item(), x.size(0))
                
                losses.update(loss_.item(), x.size(0))
                loss_.backward()
                optimizer.step()

            train_loss = losses.avg
            epoch_loss = (loss_dict['angle'].avg, loss_dict['vel'].avg, loss_dict['kl'].avg)
            losses.reset()

            # eval
            model.eval()
            for i, (x, y) in enumerate(tqdm(test_)):
                loss_dict = {'angle':utils.AverageMeter(), 'vel':utils.AverageMeter(), 'kl':utils.AverageMeter()}
                x_np = x.numpy()
                x = x.to(DEVICE)
                out, mean, log_var = model(x, self.inp_len+self.out_len, self.inp_len+self.out_len)
                
                loss_angle, loss_vel, loss_kl = self.total_loss(x_np, out, y.to(DEVICE), self.inp_len+self.out_len, mean, log_var)
                loss_ = loss_angle + loss_vel +loss_kl
                
                loss_dict['angle'].update(loss_angle.item(), x.size(0))
                loss_dict['vel'].update(loss_vel.item(), x.size(0))
                loss_dict['kl'].update(loss_kl.item(), x.size(0))
                
                losses.update(loss_.item(), x.size(0))

            # save
            if losses.avg < best_loss:
                torch.save(model, model_path + "/best.pth")
                best_loss = losses.avg
            else:
                torch.save(model, model_path + "/last.pth")
            result = "Part = {}, Epoch = {:3}/{}, train_loss = {:10}, test_loss = {:10}".format(part, epoch+1, epochs, round(train_loss, 7), round(losses.avg, 7))
            loss_list.append([round(train_loss, 7), round(losses.avg, 7)])
            loss_detail.append([round(epoch_loss[0],7), round(epoch_loss[1],7), round(epoch_loss[2],7), round(loss_dict['angle'].avg, 7), round(loss_dict['vel'].avg, 7), round(loss_dict['vel'].avg, 7)])
            print('Using device:', DEVICE)
            print(result)
            self.save_result(model_path, result)
            losses.reset()
        with open(model_path + '/loss.pkl', 'wb')as fpick:
            pickle.dump(loss_list, fpick)   
        with open(model_path + '/loss_detail.pkl', 'wb')as fpick:
            pickle.dump(loss_detail, fpick)        
        print("done!")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, help="Dataset Dir", default="Human3.6M")
    parser.add_argument("-r", "--train_dir", type=str, help="Train directory", default="train_angle")
    parser.add_argument("-c", "--train_ca", type=str, help="Train class")
    parser.add_argument("-i", "--inp_len", type=int, help="Input length", default=20)
    parser.add_argument("-o", "--out_len", type=int, help="Output length", default=10)
    parser.add_argument("-v", "--version", type=str, help="Joint definition", default="V3")
    parser.add_argument("-p", "--prefix", type=str, help="Model path prefix", default="test")
    args = parser.parse_args()

    save_path = os.path.join("ckpt", f"{args.prefix}_{args.version}_{args.dataset}_{args.train_dir}_{args.train_ca}_{args.inp_len}{args.out_len}")

    batch_size = 128
    epochs = 250
    lr = 0.0001
    hyperparams = {}
    hyperparams['batch_size'] = batch_size
    hyperparams['epochs'] = epochs
    hyperparams['lr'] = lr

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        opt = {"dataset":args.dataset, "train_dir":args.train_dir, "train_ca":args.train_ca, 
                "lr":lr, "batch_size":batch_size, "epochs":epochs, "inp_len":args.inp_len, "out_len":args.out_len}
        with open(save_path + '/opt.json', 'w') as fp:
            json.dump(opt, fp)
    
    # joint_def = JointDefV3()
    joint_ver = "JointDef" + args.version
    mod = __import__('model_joints', fromlist=joint_ver)
    joint_def = getattr(mod, joint_ver)
    joint_def = joint_def()

    train = Train(joint_def, args.dataset, args.train_dir, args.train_ca, 
                    args.inp_len, args.out_len, hyperparams)
    train_, test_, train_sampler, test_sampler = train.load_data()

    # train fullbody
    # part = 'entire'
    # model = load_model(part)
    # train(model, part, train_, test_)

    # train part
    for part in joint_def.part_list:
        model = train.load_model(part)
        train_, test_ = train.divide_data(train_sampler, test_sampler, part)
        train.train(model, part, train_, test_, save_path)  
