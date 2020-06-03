import torch
from torch import nn
import sys
from torch.nn import functional as F
from IncrementalLeraningMLDL.src.Exemplars import Exemplars

class Icarl(Exemplars):
    def __init__(self,K=2000):
       super.__init__(K)
       
    def predict(self,images,net):
        """
        Args: 
        - images: batch di immagini da predirre
        - net : rete

        Returns: 
        - lista di predizioni per quel batch
        """
        exemplar_means = self.exemplar_centroids
        means = torch.stack(exemplar_means) # (n_classes, feature_size)
        means = torch.stack([means] * len(images)) # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2) # (batch_size, feature_size, n_classes)
        
        with torch.no_grad():
            net.train(False)
            features = net.feature_extractor(images) # (batch_size, feature_size)
            for i in range(features.size(0)): 
                features.data[i] = features.data[i] / features.data[i].norm() # Normalize
            features = features.unsqueeze(2) # (batch_size, feature_size, 1)
            features = features.expand_as(means) # (batch_size, feature_size, n_classes)

            dists = (features - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
            _, preds = dists.min(1)

        return preds
    
    def compute_loss(self,old_outputs,new_output,labels,step,n_classes,current_step,utils):
        sigmoid = nn.Sigmoid()
        n_old_classes = n_classes*(step-1)
        clf_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')
        dist_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')
        
        if step == 1 or current_step==-1:
            clf_loss = clf_criterion(new_output,utils.one_hot_matrix(labels,n_classes*step))
            return clf_loss,clf_loss,clf_loss-clf_loss
        clf_loss = clf_criterion(new_output[:,n_old_classes:],utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:])
        dist_loss = dist_criterion(new_output[:,:n_old_classes],sigmoid(old_outputs))
        
        targets = utils.one_hot_matrix(labels,n_classes*step)
        targets[:,:n_old_classes] = sigmoid(old_outputs)
        tot_loss = clf_criterion(new_output,targets)


        return tot_loss,clf_loss*1/step,dist_loss*(step-1)/step

    def compute_loss_L2(self,old_outputs,new_output,labels,step,n_classes,current_step,utils):
        sigmoid = nn.Sigmoid()
        n_old_classes = n_classes*(step-1)
        clf_criterion = nn.MSELoss(reduction='mean')
        dist_criterion = nn.MSELoss(reduction='mean')
        
        if step == 1 or current_step==-1:
            clf_loss = clf_criterion(sigmoid(new_output),utils.one_hot_matrix(labels,n_classes*step))
            return clf_loss,clf_loss,clf_loss-clf_loss
        clf_loss = clf_criterion(sigmoid(new_output[:,n_old_classes:]),utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:])
        dist_loss = dist_criterion(sigmoid(new_output[:,:n_old_classes]),sigmoid(old_outputs))
        
        targets = utils.one_hot_matrix(labels,n_classes*step)
        targets[:,:n_old_classes] = sigmoid(old_outputs)
        tot_loss = clf_criterion(sigmoid(new_output),targets)


        return tot_loss,clf_loss*1/step,dist_loss*(step-1)/step


    def compute_loss_paper():
            pass