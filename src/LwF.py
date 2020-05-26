from torch import nn
class LwF():
    def __init__(self):
        pass

    def compute_loss(self,old_outputs,new_output,labels,step,n_classes,current_step,utils):
        sigmoid = nn.Sigmoid()
        n_old_classes = n_classes*(step-1)
        clf_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')
        dist_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')
        
        if step == 1 or current_step == -1:
            clf_loss = clf_criterion(new_output,utils.one_hot_matrix(labels,n_classes*step))
            return clf_loss,clf_loss,clf_loss-clf_loss
        clf_loss = clf_criterion(new_output[:,n_old_classes:],utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:])
        dist_loss = dist_criterion(new_output[:,:n_old_classes],sigmoid(old_outputs))
        
        targets = utils.one_hot_matrix(labels,n_classes*step)
        targets[:,:n_old_classes] = sigmoid(old_outputs)
        tot_loss = clf_criterion(new_output,targets)


        return tot_loss,clf_loss*1/step,dist_loss*(step-1)/step
    