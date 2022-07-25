import numpy as np
import copy 

import torch 
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.data import DatasetSplit 
from ..pruning.unstructured import *
from ..ours.our_algorithm_utils import *
from ..fedspa.rigil import *

class Client_Sub_Un(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device, mask, pruning_target, 
                 train_ds=None, train_idxs=None, test_ds = None, test_idxs = None,val_ds=None, val_idxs=None):
        
        self.name = name 
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr 
        self.momentum = momentum 
        self.device = device 
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(train_ds, train_idxs), batch_size=self.local_bs, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(test_ds, test_idxs), batch_size=200)
        self.ldr_val = DataLoader(DatasetSplit(val_ds, val_idxs), batch_size=200)
        self.mask = mask 
        self.pruning_target = pruning_target
        self.acc_best = 0 
        self.count = 0 
        self.pruned = 0 
        self.save_best = True 
        self.fake_net = copy.deepcopy(model) # For local training model to calculate weight divergence
        
    def train(self, percent, dist_thresh, acc_thresh, is_print = False):
        self.net.to(self.device)
        self.net.train()
        
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
        
        epoch_loss = []
        m1 = copy.deepcopy(self.mask)
        m2 = copy.deepcopy(self.mask)
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                
                # Freezing Pruned weights by making their gradients Zero
                step = 0
                for name, p in self.net.named_parameters():
                    if 'weight' in name:
                        tensor = p.data.cpu().numpy()
                        grad_tensor = p.grad.data.cpu().numpy()
                        #grad_tensor = np.where(tensor < EPS, 0, grad_tensor)
                        grad_tensor = grad_tensor * self.mask[step]
                        p.grad.data = torch.from_numpy(grad_tensor).to(self.device)
                        step = step + 1 
                        
                optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
            if iteration+1 == 1: 
                m1 = fake_prune(percent, copy.deepcopy(self.net), copy.deepcopy(self.mask))
            if iteration+1 == 5:     
                m2 = fake_prune(percent, copy.deepcopy(self.net), copy.deepcopy(self.mask))
        
        if self.save_best: 
            _, acc = self.eval_test()
            if acc > self.acc_best:
                self.acc_best = acc 
                
        dist = dist_masks(m1, m2)
        if is_print:
            print(f'Distance: {dist}')
        
        state_dict = copy.deepcopy(self.net.state_dict())
        final_mask = copy.deepcopy(self.mask)
        
        _,val_acc = self.eval_val()
        if dist > dist_thresh and self.pruned < self.pruning_target and val_acc > acc_thresh: 
            if (self.pruning_target - self.pruned < percent): 
                print(f'..IMPOSING PRUNING To Reach Target PRUNING..')
                #print(f'user prune: {user_pruned}')
                percent = ((((100 - self.pruned) - (100 - self.pruning_target))/(100 - self.pruned)) * 100)
                #print(f'Percent {percent}')
                if percent > 5: 
                    percent = 5
                m2 = fake_prune(percent, copy.deepcopy(self.net), copy.deepcopy(self.mask))

            old_dict = copy.deepcopy(self.net.state_dict())
            new_dict = real_prune(copy.deepcopy(self.net), m2)
            self.net.load_state_dict(new_dict)
            _, acc = self.eval_test()
            if is_print:
                print(f'acc after pruning: {acc}')
            if acc > acc_thresh: 
                if is_print:
                    print(f'Pruned! acc after pruning {acc}')
                state_dict = new_dict 
                final_mask = m2 
            else: 
                if is_print:
                    print(f'Not Pruned!!!')
                state_dict = old_dict 
                final_mask = copy.deepcopy(self.mask)
                
        self.net.load_state_dict(state_dict)
        self.mask = copy.deepcopy(final_mask) 
        self.pruned, _ = print_pruning(copy.deepcopy(self.net), is_print)
        
        return sum(epoch_loss) / len(epoch_loss)

    def fake_train(self):
        self.fake_net.to(self.device)
        self.fake_net.train()
        
        optimizer = torch.optim.SGD(self.fake_net.parameters(), lr=self.lr, momentum=self.momentum)
        
        for iteration in range(self.local_ep):
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.fake_net.zero_grad()
                optimizer.zero_grad()
                log_probs = self.fake_net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                        
                optimizer.step()

    def local_train(self):
        self.net.to(self.device)
        self.net.train()
        
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
        
        for iteration in range(self.local_ep):
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                        
                optimizer.step()


    
    def get_state_dict(self):
        return self.net.state_dict()
    def get_fake_state_dict(self):
        return self.fake_net.state_dict()
    def get_mask(self):
        return self.mask 
    def get_best_acc(self):
        return self.acc_best
    def get_pruning(self):
        return self.pruned
    def get_count(self):
        return self.count
    def get_net(self):
        return self.net
    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)
    def set_mask(self,mask):
        self.mask = mask
    def set_pruned(self,pruned):
        self.pruned = pruned

    def update_weights(self):
        new_dict = real_prune(copy.deepcopy(self.net), copy.deepcopy(self.mask))
        self.net.load_state_dict(new_dict)

    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy
    
    def eval_train(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy

    def eval_val(self):
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_val:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                train_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_val.dataset)
        accuracy = 100. * correct / len(self.ldr_val.dataset)
        return train_loss, accuracy


    def new_algorithm_client_update(self,iteration,delta_r,alpha,T_end,mask_list,selected_idx_list,n_conv_layer,acc_thresh):

        def cosine_annealing(alpha,iteration,T_end):
            return alpha / 2 * (1 + np.cos((iteration * np.pi) / T_end))

        self.pruned, _ = print_pruning(copy.deepcopy(self.net), is_print = True)

        if (self.pruned >= self.pruning_target-1) and(iteration%delta_r == 0):
                mask_readjustment_rate = cosine_annealing(alpha,iteration,T_end)
                print("mask_readjustment_rate = ",mask_readjustment_rate)
                if mask_readjustment_rate != 0:
                    # Regrowth based on affinity matrix
                    self.mask = regrowth_based_on_affinity_c_idxs(self.mask,mask_list,selected_idx_list,mask_readjustment_rate/2,n_conv_layer)

                    # Regrowth randomly
                    next_mask_adjustment_rate = (1+mask_readjustment_rate)/(1+mask_readjustment_rate/2) - 1
                    self.mask,self.pruned,next_prune_rate = model_growing(self.mask,next_mask_adjustment_rate,n_conv_layer)

        new_dict = real_prune(copy.deepcopy(self.net), copy.deepcopy(self.mask))
        self.net.load_state_dict(new_dict)
        self.local_train()
        _,val_acc = self.eval_val()
        loss,_ = self.eval_test()

        prune = False
        if (val_acc > acc_thresh) and (self.pruned < self.pruning_target-1):
            prune = True

        weights_list = []
        for tensor in self.get_state_dict().items():
            
            weights_list.append(tensor[1])
        

        return loss, weights_list,prune

    def fedspa_client_update(self,pruner_state_dict,pruning_target,T_end,alpha,pruner=None, is_print = False):
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
        if pruner == None:
            pruner = RigLScheduler(self.net,
                                    optimizer,
                                    dense_allocation=1-pruning_target/100,
                                    sparsity_distribution="ERK",
                                    T_end=T_end,
                                    delta=self.local_ep,
                                    alpha=alpha,
                                    state_dict=pruner_state_dict)

        self.net.to(self.device)
        self.net.train()
        
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
        
        w_0 = self.get_state_dict()
        # print("w_0[fc1.weight] = ",w_0["fc1.weight"])
        for iteration in range(self.local_ep):
            
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                optimizer.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()

                if pruner():
                        
                    optimizer.step()

        self.set_mask(pruner.get_mask())
        # print("updated_mask = ",pruner.get_mask()[0][0])

        w_1 = self.get_state_dict()
        # print("w_1[fc1.weight] =",w_1["fc1.weight"])
        U_t = copy.deepcopy(w_1)

        for key,tensor in U_t.items():
            U_t[key] = tensor - w_0[key]
        loss,_ = self.eval_test()
        self.pruned, _ = print_pruning(copy.deepcopy(self.net), is_print)
        pruner_state_dict = pruner.state_dict()
        
        return loss,U_t,pruner_state_dict,pruner



                
