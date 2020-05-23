class Utils():
    def __init__(self):
        pass

    def create_dataloaders(self,training_set,test_set,group,BATCH_SIZE):
        train,val = training_set.get_single_train_joint_validation(group)
        test = test_set.get_groups_joint(group)
        
            
        train_dataloader =  DataLoader(train,batch_size=BATCH_SIZE,drop_last=True,num_workers=4,shuffle=True)
        val_dataloader = DataLoader(val,batch_size=BATCH_SIZE,drop_last=False,num_workers=4)
        test_dataloader = DataLoader(test,batch_size=BATCH_SIZE,drop_last=False,num_workers=4)

        return train_dataloader,val_dataloader,test_dataloader
    
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