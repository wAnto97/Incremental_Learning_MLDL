from torch import nn
import torch
class Loss():
    def __init__(self):
        pass
    
    def icarl_loss(self,old_outputs,new_output,labels,step,current_step,utils,n_classes=10):
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

    def L2_loss(self,old_outputs,new_output,labels,step,current_step,utils,n_classes=10):
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
    
    def abs_log_loss(self,old_outputs,new_output,labels,step,n_classes,current_step,utils):
        sigmoid = nn.Sigmoid()
        n_old_classes = n_classes*(step-1)
        clf_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')
        EPS = 3.720076e-44

        if step == 1 or current_step==-1:
            clf_loss = clf_criterion(new_output,utils.one_hot_matrix(labels,n_classes*step))
            return clf_loss,clf_loss,clf_loss-clf_loss

        clf_loss = clf_criterion(new_output[:,n_old_classes:],utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:])
        dist_loss = torch.mean(- torch.log(EPS + 1 - torch.abs(sigmoid(new_output[:,:n_old_classes]) - sigmoid(old_outputs))))
        
        tot_loss = clf_loss*1/step + dist_loss*(step-1)/step

        return tot_loss,clf_loss*1/step,dist_loss*(step-1)/step
    
    
    def abs_log_loss2(self,old_outputs,new_output,labels,step,n_classes,current_step,utils):
        sigmoid = nn.Sigmoid()
        n_old_classes = n_classes*(step-1)
        clf_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')
        dist_2_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')
        EPS = 3.720076e-44

        if step == 1 or current_step==-1:
            clf_loss = clf_criterion(new_output,utils.one_hot_matrix(labels,n_classes*step))
            return clf_loss,clf_loss,clf_loss-clf_loss

        clf_loss = clf_criterion(new_output[:,n_old_classes:],utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:])
        dist_loss = torch.mean(- torch.log(EPS + 1 - torch.abs(sigmoid(new_output[:,:n_old_classes]) - sigmoid(old_outputs))))
        dist_2_loss = dist_2_criterion(new_output[:,:n_old_classes],sigmoid(old_outputs))

        tot_loss = clf_loss*1/step + (0.5*dist_loss + 0.5*dist_2_loss)*(step-1)/step

        return tot_loss,clf_loss*1/step,(0.5*dist_loss + 0.5*dist_2_loss)*(step-1)/step


    def MMLoss(self,old_outputs,new_output,labels,step,n_classes,current_step,utils):
        sigmoid = nn.Sigmoid()
        n_old_classes = n_classes*(step-1)
        
        if step == 1 or current_step==-1:
            y = utils.one_hot_matrix(labels,n_classes*step)
            clf_loss = torch.mean(-1/4* (4*(2*y - 1).pow(3) * (2*sigmoid(new_output) - 1) - (2*sigmoid(new_output) - 1).pow(4) - 3))
            return clf_loss,clf_loss,clf_loss-clf_loss 


        y = utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:]
        clf_loss = torch.mean(-1/4* (4*(2*y - 1).pow(3) * (2*sigmoid(new_output[:,n_old_classes:]) - 1) - (2*sigmoid(new_output[:,n_old_classes:]) - 1).pow(4) - 3))
       
        target = sigmoid(old_outputs)
        dist_loss = torch.mean(-1/4* (4*(2*target - 1).pow(3) * (2*sigmoid(new_output[:,:n_old_classes]) - 1) - (2*sigmoid(new_output[:,:n_old_classes]) - 1).pow(4) - 3))
       
        tot_loss = clf_loss*1/step + dist_loss*(step-1)/step
        return tot_loss,clf_loss*1/step,dist_loss*(step-1)/step

    def MMLoss_onlydist(self,old_outputs,new_output,labels,step,n_classes,current_step,utils):
        sigmoid = nn.Sigmoid()
        n_old_classes = n_classes*(step-1)
        clf_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')

        if step == 1 or current_step==-1:
            clf_loss = clf_criterion(new_output,utils.one_hot_matrix(labels,n_classes*step))
            return clf_loss,clf_loss,clf_loss-clf_loss

        clf_loss = clf_criterion(new_output[:,n_old_classes:],utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:])
        target = sigmoid(old_outputs)
        dist_loss = torch.mean(-1/4* (4*(2*target - 1).pow(3) * (2*sigmoid(new_output[:,:n_old_classes]) - 1) - (2*sigmoid(new_output[:,:n_old_classes]) - 1).pow(4) - 3))
       
        tot_loss = clf_loss*1/step + dist_loss*(step-1)/step
        return tot_loss,clf_loss*1/step,dist_loss*(step-1)/step

    def MMLoss_CE(self,old_outputs,new_output,labels,step,n_classes,current_step,utils):
        sigmoid = nn.Sigmoid()
        n_old_classes = n_classes*(step-1)
        clf_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')

        if step == 1 or current_step==-1:
            clf_loss = clf_criterion(new_output,utils.one_hot_matrix(labels,n_classes*step))
            return clf_loss,clf_loss,clf_loss-clf_loss

        clf_loss = clf_criterion(new_output[:,n_old_classes:],utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:])
        clf_old = torch.mean(-(utils.one_hot_matrix(labels,n_classes*step)[:,:n_old_classes] * torch.log(sigmoid(new_output[:,:n_old_classes]))))
        
        target = sigmoid(old_outputs)
        dist_loss = torch.mean(-1/4* (4*(2*target - 1).pow(3) * (2*sigmoid(new_output[:,:n_old_classes]) - 1) - (2*sigmoid(new_output[:,:n_old_classes]) - 1).pow(4) - 3))
       
        tot_loss = clf_loss*1/step + (dist_loss + clf_old)*(step-1)/step
        return tot_loss,clf_loss*1/step,dist_loss*(step-1)/step


    def hybrid_loss(self,old_outputs,new_output,labels,step,n_classes,current_step,utils):
        sigmoid = nn.Sigmoid()
        n_old_classes = n_classes*(step-1)
        clf_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')
        dist_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')
        
        if step == 1 or current_step==-1:
            clf_loss = clf_criterion(new_output,utils.one_hot_matrix(labels,n_classes*step))
            return clf_loss,clf_loss,clf_loss-clf_loss
        
        clf_loss = clf_criterion(new_output[:,n_old_classes:],utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:])
        dist_loss = dist_criterion(new_output[:,:n_old_classes],sigmoid(old_outputs))
        target = sigmoid(old_outputs)
        dist_MM_loss = torch.mean(-1/4* (4*(2*target - 1).pow(3) * (2*sigmoid(new_output[:,:n_old_classes]) - 1) - (2*sigmoid(new_output[:,:n_old_classes]) - 1).pow(4) - 3))

        tot_loss = clf_loss*1/step + (0.9 * dist_MM_loss + 0.1 * dist_loss)*(step-1)/step

        return tot_loss,clf_loss*1/step,dist_loss*(step-1)/step


    def BCE_by_hand(self,old_outputs,new_output,labels,step,n_classes,current_step,utils):
        sigmoid = nn.Sigmoid()
        EPS = 3.720076e-44
        n_old_classes = n_classes*(step-1)

        y = utils.one_hot_matrix(labels,n_classes*step)
        if step == 1 or current_step==-1:
          #clf = torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(new_output, y))
          clf_loss = torch.mean(-(y*torch.log(sigmoid(new_output)+EPS) + (1-y)* torch.log(1 - sigmoid(new_output)+EPS)))

          return clf_loss, clf_loss,clf_loss-clf_loss
        
        y_2 = utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:]
        clf_loss = torch.mean(-(y_2*torch.log(sigmoid(new_output[:,n_old_classes:])+EPS) + (1-y_2)* torch.log(1 - sigmoid(new_output[:,n_old_classes:])+EPS)))
        
        target = sigmoid(old_outputs)
        dist_loss = torch.mean(-(target*torch.log(sigmoid(new_output[:,:n_old_classes])+EPS) + (1-target)* torch.log(1 - sigmoid(new_output[:,:n_old_classes])+EPS)))
        
        tot_loss = clf_loss*1/step + dist_loss*(step-1)/step

        return tot_loss,clf_loss*1/step,dist_loss*(step-1)/step