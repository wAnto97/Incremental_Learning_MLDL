import json
import yaml
import torch
from torch.utils.data import  DataLoader

class Utils():
    def __init__(self):
        pass

    def writeOnFileLosses(self,file_path,group,losses):
        training_loss = losses[0]
        validation_loss = losses[1]

        json_out = {}
        json_out['group'+str(group)] = {}
        json_out['group'+str(group)]['training_loss'] = training_loss
        json_out['group'+str(group)]['validation_loss'] = validation_loss

        with open(file_path,mode='a') as file_out:
            json.dump(json_out,file_out)
            file_out.write('\n')

    def writeOnFileMetrics(self,file_path,group,metrics):
        training_accuracy = metrics[0]
        validation_accuracy = metrics[1]
        test_accuracy = metrics[2]
        

        json_out = {}
        json_out['group'+str(group)] = {}
        json_out['group'+str(group)]['training_accuracy'] = training_accuracy
        json_out['group'+str(group)]['validation_accuracy'] = validation_accuracy
        json_out['group'+str(group)]['test_accuracy'] = test_accuracy
        if(len(metrics) == 4):
            conf_matrix = metrics[3]
            json_out['group'+str(group)]['conf_matrix'] = conf_matrix

        with open(file_path,mode='a') as file_out:
            json.dump(json_out,file_out)
            file_out.write('\n')

    def readFileLosses(self,file_path,type):
        if type == 'LwF' or type == 'iCarl':
            loss1 = 'classification_loss'
            loss2 = 'distillation_loss'
        else:
            loss1 = 'training_loss'
            loss2 = 'validation_loss'

        val_losses_per_group = []
        train_loss_per_group = []
        with open(file_path,mode='r') as f:
            for index,line in enumerate(f):
                json_obj = (yaml.load(str(line)))
                train_loss_per_group.append(json_obj['group' + str(index+1)][loss1])
                val_losses_per_group.append(json_obj['group' + str(index+1)][loss2])
        
        return {'train_losses':train_loss_per_group,'validation_losses' :val_losses_per_group}

    def readFileMetrics(self,file_path,cm=False):
        accuracy_train_per_group = []
        accuracy_val_per_group = []
        accuracy_test_per_group = []
        cm_per_group = []
        
        with open(file_path,mode='r') as f:
            for index,line in enumerate(f):
                json_obj = (yaml.load(str(line)))
                accuracy_train_per_group.append(json_obj['group' + str(index+1)]['training_accuracy'])
                accuracy_val_per_group.append(json_obj['group' + str(index+1)]['validation_accuracy'])
                accuracy_test_per_group.append(json_obj['group' + str(index+1)]['test_accuracy'])
                if cm == True:
                    cm_per_group.append(json_obj['group' + str(index+1)]['conf_matrix'])
            
        if(cm==True):
            return {'accuracy_train' : accuracy_train_per_group,
                'accuracy_val_per_group' : accuracy_val_per_group,
                'accuracy_test_per_group' : accuracy_test_per_group,
                'conf_matrix' : cm_per_group
                }


        return {'accuracy_train':accuracy_train_per_group,
                'accuracy_val_per_group' : accuracy_test_per_group,
                'accuracy_test_per_group' : accuracy_test_per_group,
                }

    def create_dataloaders(self,training_set,test_set,group,BATCH_SIZE):
        train,val = training_set.get_single_train_joint_validation(group)
        test = test_set.get_groups_joint(group)
        
            
        train_dataloader =  DataLoader(train,batch_size=BATCH_SIZE,drop_last=True,num_workers=4,shuffle=True)
        val_dataloader = DataLoader(val,batch_size=BATCH_SIZE,drop_last=False,num_workers=4)
        test_dataloader = DataLoader(test,batch_size=BATCH_SIZE,drop_last=False,num_workers=4)

        return train_dataloader,val_dataloader,test_dataloader
    
    def create_dataloaders_icarl(self, training_set,test_set,group,exemplar_set,BATCH_SIZE):
        train=training_set.get_train_exemplars(group,exemplar_set)
        test = test_set.get_groups_joint(group)

        train_dataloader =  DataLoader(train,batch_size=BATCH_SIZE,drop_last=True,num_workers=4,shuffle=True)
        test_dataloader = DataLoader(test,batch_size=BATCH_SIZE,drop_last=False,num_workers=4)

        return train_dataloader,test_dataloader

    def create_dataloaders_CB(self, training_set,test_set,group,exemplar_set,BATCH_SIZE):
       train,val = training_set.get_train_val_CB(group,exemplar_set)
       test = test_set.get_groups_joint(group)

       train_dataloader =  DataLoader(train,batch_size=BATCH_SIZE,drop_last=True,num_workers=4,shuffle=True)
       val_dataloader = DataLoader(val,batch_size=int(BATCH_SIZE/2),drop_last=True,num_workers=4,shuffle=True)
       test_dataloader = DataLoader(test,batch_size=BATCH_SIZE,drop_last=False,num_workers=4)

       return train_dataloader,val_dataloader,test_dataloader


    def create_dataloaders_icarl_validation(self, training_set,test_set,group,exemplar_set,BATCH_SIZE):
        train=training_set.get_train_exemplars(group,exemplar_set)
        test = test_set.get_groups_joint(group)

        train_dataloader =  DataLoader(train,batch_size=BATCH_SIZE,drop_last=True,num_workers=4,shuffle=True)
        test_dataloader = DataLoader(test,batch_size=BATCH_SIZE,drop_last=False,num_workers=4)

        return train_dataloader,test_dataloader
    
    def create_onehot(self,intLabel,num_classes):
        onehot = torch.zeros(num_classes)
        onehot[intLabel]=1
        return onehot.cuda()

    def one_hot_matrix(self,labels,n_classes):
        matrix = torch.zeros((len(labels),n_classes))
        for index,y in enumerate(labels):
            matrix[index] = self.create_onehot(y,n_classes)
        return matrix.cuda()

    def create_images_indices(self,dataloader,step,n_classes=10):
        images_indices=[]
        for i in range(n_classes):
            images_indices.append([])
        for images,labels,indices in dataloader:
            for img,label,index in zip(images,labels,indices):
                images_indices[label-(step-1)*n_classes].append((img,index))

        return images_indices
