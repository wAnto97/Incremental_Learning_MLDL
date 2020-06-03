from IncrementalLeraningMLDL.src.resnet import resnet32
from IncrementalLeraningMLDL.src.resnet_cosine import resnet32 as resnet32_cosine
import torch
from torch import nn
import copy
import torch.optim as optim

class MyNet():
    def __init__(self,n_classes,type='normal'):
        self.type = type
        if(type == 'normal'):
          self.net = resnet32(num_classes=10)
        elif type == 'cosine':
            self.net = resnet32_cosine(num_classes=10)
        self.net.linear = nn.Linear(64,n_classes)
        self.init_weights = torch.nn.init.kaiming_normal_(self.net.linear.weight)
        self.batch_classes = 10
        self.prev_net = None
    
    def update_network(self,best_net,n_classes,init_weights):
        self.prev_net = copy.deepcopy(best_net)
        prev_weights = copy.deepcopy(best_net.linear.weight)
        prev_bias = copy.deepcopy(best_net.linear.bias)
        self.net.linear = nn.Linear(64,n_classes)
        self.net.linear.weight.data[:n_classes-self.batch_classes] = prev_weights
        self.net.linear.bias.data[:n_classes-self.batch_classes] = prev_bias

        return self.prev_net,self.net

    def get_old_outputs(self,images,labels):
        self.prev_net.train(False)
        output = self.prev_net(images.cuda())
        return output

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
        for i,child in enumerate(self.net.children()):
            if(i==5):
                break
            for param in child.parameters():
                param.requires_grad = True
        
        return self.net