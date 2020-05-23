from IncrementalLeraningMLDL.resnet_cifar import resnet32
import torch
from torch import nn
import copy


class MyNet():
    def __init__(self,n_classes):
        self.net = resnet32()
        self.net.linear = nn.Linear(64,n_classes)
        self.init_weights = torch.nn.init.kaiming_normal_(self.net.linear.weight)
    
    def update_network(self,best_net,n_classes,init_weights):
        prev_net = copy.deepcopy(best_net)
        n_old_classes = best_net.linear.weight.shape[0]
        prev_weights = copy.deepcopy(best_net.linear.weight)
        self.net.linear = nn.Linear(64,n_classes)
        self.net.linear.weight.data = torch.cat((prev_weights,init_weights))

        return prev_net,self.net

    def prepare_training(self,LR,MOMENTUM,WEIGHT_DECAY,STEP_SIZE,GAMMA):    
        parameters_to_optimize = self.net.parameters()
        optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, STEP_SIZE, gamma=GAMMA)

        return (optimizer,scheduler)