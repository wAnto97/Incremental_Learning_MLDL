from torch import nn
import torch
import numpy as np

class Loss():
    def __init__(self):
        pass
    
    def icarl_loss(self,old_outputs,new_output,labels,step,current_step,utils,n_classes=10):
        '''BCE loss. Citata nel paper di iCarl'''

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
    def LfC_loss(self,old_outputs,new_features,new_output,labels,step,current_step,utils,eta,lambda_base = 10,n_classes=10,batch_size=128,m=0.5,K=2):
        n_old_classes = n_classes*(step-1)
        clf_criterion = nn.CrossEntropyLoss(reduction = 'mean')
        cosine_loss = nn.CosineEmbeddingLoss(reduction='mean')
        if step == 1 or current_step==-1:
            clf_loss = clf_criterion(new_output,labels)
            return clf_loss
        
        #Classification
        clf_loss = clf_criterion(new_output,labels)
        
        #Distillation
        lambda_dist = lambda_base*((n_classes/n_old_classes)**0.5)
        dist_loss = lambda_dist*cosine_loss(new_features, old_outputs,torch.ones(batch_size).cuda()) 

        #Margin
        exemplar_idx = sum(labels.cpu().numpy() == label for label in range(n_old_classes)).astype(bool)
        exemplar_labels = labels[exemplar_idx].type(torch.long)
        anchors = new_output[exemplar_idx, exemplar_labels] / eta
        out_new_classes = new_output[exemplar_idx, n_old_classes:] / eta
        topK_hard_negatives, _ = torch.topk(out_new_classes, K)
        loss_mr = torch.max(m - anchors.unsqueeze(1).cuda() + topK_hard_negatives.cuda(), torch.zeros(1).cuda()).sum(dim=1).mean()


        return clf_loss + dist_loss + loss_mr


    def L2_loss(self,old_outputs,new_output,labels,step,current_step,utils,n_classes=10):
        '''MSE loss. Comportamento peggiore rispetto alla BCE (influenzato dalla scelta di parametri non ottimali).'''
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


    def MMLoss(self,old_outputs,new_output,labels,step,current_step,utils,n_classes=10, w=1/4):
        '''
        Funzione di loss polinomiale, ricavata seguendo criteri specificati all'interno del paper, con lo scopo
        di migliorare le performance della BCE loss e avere maggiore flessibilità. Essendo polinomiale, infatti,
        è possibile cambiarne il comportamento agendo sui coefficienti, avendo cura di apportare modifiche rispettando
        sempre i criteri prima citati. E' pensata per essere usata soltanto come termine di distillation, insieme alla BCE
        per la classificazione (vedi MMLoss_onlydist).
        Comportamento: Favorisce la modifica di probabilità che al tempo 't-1' erano prossime a 0.5, 
        rispetto a quelle prossime a zero o uno
        '''
        # WARNING: Usare MMLoss_onlydist
        sigmoid = nn.Sigmoid()
        n_old_classes = n_classes*(step-1)
        w_clf = 3/160 # Peso inserito in modo tale che il Learning Rate sia sempre vicino a 2. Così è più semplice trovare i parametri esatti.
        
        if step == 1 or current_step==-1:
            y = utils.one_hot_matrix(labels,n_classes*step)
            clf_loss = torch.mean(-w_clf*(4*(2*y - 1).pow(3) * (2*sigmoid(new_output) - 1) - (2*sigmoid(new_output) - 1).pow(4) - 3))
            return clf_loss,clf_loss,clf_loss-clf_loss 


        y = utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:]
        clf_loss = torch.mean(-w_clf* (4*(2*y - 1).pow(3) * (2*sigmoid(new_output[:,n_old_classes:]) - 1) - (2*sigmoid(new_output[:,n_old_classes:]) - 1).pow(4) - 3))
       
        target = sigmoid(old_outputs)
        dist_loss = torch.mean(-w_clf* (4*(2*target - 1).pow(3) * (2*sigmoid(new_output[:,:n_old_classes]) - 1) - (2*sigmoid(new_output[:,:n_old_classes]) - 1).pow(4) - 3))
       
        tot_loss = clf_loss*1/step + dist_loss*(step-1)/step
        return tot_loss,clf_loss*1/step,dist_loss*(step-1)/step

    def MMLoss_onlydist(self,old_outputs,new_output,labels,step,current_step,utils,n_classes=10, w=1/4):
        '''
        w serve ad allineare i contributi di classification e distillation in modo tale che abbiano pendenze e learning rate simili.
        senza questo fattore la distillation avrebbe un peso molto maggiore rispetto alla clf. se si usa la BCE impostare w=1/4 (o 1/3)
        Il valore di default è stato trovato usando un approccio grafico
        '''
        sigmoid = nn.Sigmoid()
        n_old_classes = n_classes*(step-1)
        clf_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')

        if step == 1 or current_step==-1:
            clf_loss = clf_criterion(new_output,utils.one_hot_matrix(labels,n_classes*step))
            return clf_loss,clf_loss,clf_loss-clf_loss

        clf_loss = clf_criterion(new_output[:,n_old_classes:],utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:])
        target = sigmoid(old_outputs)
        dist_loss = torch.mean(- w * (4*(2*target - 1).pow(3) * (2*sigmoid(new_output[:,:n_old_classes]) - 1) - (2*sigmoid(new_output[:,:n_old_classes]) - 1).pow(4) - 3))
       
        tot_loss = clf_loss*1/step + dist_loss*(step-1)/step
        return tot_loss,clf_loss*1/step,dist_loss*(step-1)/step


    def MMLoss_onlydist_Prob(self,old_outputs,new_output,labels,step,current_step,utils, prob_vect,n_classes=10, w=1/4):
        '''
        Stessa di MM_onlydist, ma rimuove alcuni contributi di distillation, moltiplicando per un tensore random 
        (con probabilità variabile) di 0 e 1
        '''
        sigmoid = nn.Sigmoid()
        n_old_classes = n_classes*(step-1)
        clf_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')

        if step == 1 or current_step==-1:
            clf_loss = clf_criterion(new_output,utils.one_hot_matrix(labels,n_classes*step))
            return clf_loss,clf_loss,clf_loss-clf_loss

        clf_loss = clf_criterion(new_output[:,n_old_classes:],utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:])
        target = sigmoid(old_outputs)
        
        dist_loss = torch.mean(prob_vect.cuda() * (- w * (4*(2*target - 1).pow(3) * (2*sigmoid(new_output[:,:n_old_classes]) - 1) - (2*sigmoid(new_output[:,:n_old_classes]) - 1).pow(4) - 3)))
       
        tot_loss = clf_loss*1/step + dist_loss*(step-1)/step
        return tot_loss,clf_loss*1/step,dist_loss*(step-1)/step


    def MMLoss_CE(self,old_outputs,new_output,labels,step,current_step,utils,n_classes=10, w=1/4):
        '''
        w serve ad allineare i contributi di classification e distillation in modo tale che abbiano pendenze e learning rate simili.
        senza questo fattore la distillation avrebbe un peso molto maggiore rispetto alla clf. se si usa la BCE impostare w=1/4 (o 1/3)
        Il valore di default è stato trovato usando un approccio grafico
        '''
        sigmoid = nn.Sigmoid()
        n_old_classes = n_classes*(step-1)
        clf_criterion = nn.BCEWithLogitsLoss(reduction = 'mean')

        if step == 1 or current_step==-1:
            clf_loss = clf_criterion(new_output,utils.one_hot_matrix(labels,n_classes*step))
            return clf_loss,clf_loss,clf_loss-clf_loss

        clf_loss = clf_criterion(new_output[:,n_old_classes:],utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:])
        clf_old = torch.mean(-(utils.one_hot_matrix(labels,n_classes*step)[:,:n_old_classes] * torch.log(sigmoid(new_output[:,:n_old_classes]))))
        
        target = sigmoid(old_outputs)
        dist_loss = torch.mean(- w * (4*(2*target - 1).pow(3) * (2*sigmoid(new_output[:,:n_old_classes]) - 1) - (2*sigmoid(new_output[:,:n_old_classes]) - 1).pow(4) - 3))
       
        tot_loss = clf_loss*1/step + (dist_loss + clf_old)*(step-1)/step
        return tot_loss,clf_loss*1/step,dist_loss*(step-1)/step


    def MMLoss_bounded(self,old_outputs,new_output,labels,step,current_step,utils,n_classes=10):
        '''
        Combinazione lineare tra la loss polinomiale 'custom' e una BCE, con coefficienti 0.9*MM + 0.1*BCE
        Stesse caratteristiche della MMLoss, ma la funzione è limitata tra 0 e 1 come la BCE loss.
        Il comportamento rimane lo stesso della MM loss, quindi è sconveniente utilizzare questa (solo formalità)
        '''

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

    def abs_log_loss(self,old_outputs,new_output,labels,step,current_step,utils,n_classes=10):
        '''
        Utilizza come termine di distillation una funzione con pendenza costante rispetto a p(t-1).
        Di conseguenza l'incremento della loss dovuto a un cambiamento del vettore di probabilità
        dipende soltanto dal gap "p(t)-p(t-1)" e non dal valore di p(t) 
        (comportamento opposto rispetto a quello di MMloss che favorisce spostamenti intorno a 0.5).
        Il comportamento in termini di accuracy è peggiore rispetto alla MMloss. 
        '''
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
    
    
    def abs_log_loss2(self,old_outputs,new_output,labels,step,current_step,utils,n_classes=10):
        '''
        Combinazione lineare tra abs_log_loss e BCE
        '''
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


    def BCE_by_hand(self,old_outputs,new_output,labels,step,current_step,utils,n_classes=10):
        '''
        BCE loss implementata manualmente per avere un punto di riferimento per gli iperparametri e il tuning.
        '''
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


