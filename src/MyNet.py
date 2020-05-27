from IncrementalLeraningMLDL.src.resnet_cifar_c import resnet32
import torch
from torch import nn
import copy
import torch.optim as optim

class MyNet():
    def __init__(self,n_classes):
        self.net = resnet32()
        self.net.linear = nn.Linear(64,n_classes)
        self.init_weights = torch.nn.init.kaiming_normal_(self.net.linear.weight)
    
    def update_network(self,best_net,n_classes,init_weights):
        prev_net = copy.deepcopy(best_net)
        prev_weights = copy.deepcopy(best_net.linear.weight)
        self.net.linear = nn.Linear(64,n_classes)
        self.net.linear.weight.data = torch.cat((prev_weights,init_weights))

        return prev_net,self.net

    def prepare_training(self,LR,MOMENTUM,WEIGHT_DECAY,STEP_SIZE,GAMMA,typeScheduler):    
        parameters_to_optimize = self.net.parameters()
        optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        if typeScheduler == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, STEP_SIZE, gamma=GAMMA)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')

        return (optimizer,scheduler)
    
    def freeze_conv(self):
        for i,child in enumerate(self.net.children()):
            if(i==5):
                break
            for param in child.parameters():
                param.requires_grad = False
        
        return self.net

    def freeze_neurons(self,n_old_classes):
        for param in self.net.linear.parameters():
            param.grad[:n_old_classes]=0
        
        return self.net

    def unfreeze_conv(self):
        for i,child in enumerate(net.children()):
            if(i==5):
                break
            for param in child.parameters():
                param.requires_grad = True
        
        return self.net