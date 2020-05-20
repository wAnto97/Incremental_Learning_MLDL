import json
import yaml

class utils():
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
            classification_report = metrics[3]
            json_out['group'+str(group)]['report'] = classification_report

        with open(file_path,mode='a') as file_out:
            json.dump(json_out,file_out)
            file_out.write('\n')

    def readFileLosses(self,file_path):
        val_losses_per_group = []
        train_loss_per_group = []
        with open(file_path,mode='r') as f:
            for index,line in enumerate(f):
                json_obj = (yaml.load(str(line)))
                train_loss_per_group.append(json_obj['group' + str(index+1)]['training_loss'])
                val_losses_per_group.append(json_obj['group' + str(index+1)]['validation_loss'])
        
        return train_loss_per_group,val_losses_per_group

    def readFileMetrics(self,file_path,report=False):
        accuracy_train_per_group = []
        accuracy_val_per_group = []
        accuracy_test_per_group = []
        report_per_group = []
        
        with open(file_path,mode='r') as f:
            for index,line in enumerate(f):
                json_obj = (yaml.load(str(line)))
                accuracy_train_per_group.append(json_obj['group' + str(index+1)]['training_accuracy'])
                accuracy_val_per_group.append(json_obj['group' + str(index+1)]['validation_accuracy'])
                accuracy_test_per_group.append(json_obj['group' + str(index+1)]['test_accuracy'])
                if report == True:
                    report_per_group.append(json_obj['group' + str(index+1)]['report'])
            
        if(report==True):
            return accuracy_train_per_group,accuracy_val_per_group,accuracy_test_per_group,report_per_group

        return accuracy_train_per_group,accuracy_val_per_group,accuracy_test_per_group
