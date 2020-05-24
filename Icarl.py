import torch
import numpy as np
from torch import nn
import sys
class Icarl():
    def __init__(self):
        self.exemplar_set=[]
        self.exemplar_centroids=[]
        self.K=2000
    
    def build_exemplars(self,net,images_indices,n_old_classes,n_classes=10):
        m=int(self.K/n_old_classes)
        for i in range(n_classes):
            self.exemplar_set.append(self.construct_exemplar_class_set(net,images_indices[i],m))

        return self.exemplar_set
    
    def reduce_exemplars(self,n_old_classes):
        m = int(self.K/n_old_classes)
        for i in range(len(self.exemplar_set)):
            self.exemplar_set[i]=self.exemplar_set[i][:m]
        
        return self.exemplar_set
    
    def construct_exemplar_class_set(self,net, images_indices, m):
        """Construct an exemplar set for image set
        Args:
            images: np.array containing images of a class
        """
        # Compute and cache features for each example
        features = []
        for img,_ in images_indices:
            img=img.unsqueeze(0)
            feature = net.feature_extractor(img.cuda()).data.cpu().numpy() #-> la nostra feature extractor
            feature = feature / np.linalg.norm(feature) # Normalize
            features.append(feature[0])

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean) # Normalize

        inserted_indices =[]
        exemplar_class_set = []
        exemplar_features = [] # list of Variables of shape (feature_size,)

        for k in range(m):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0/(k+1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            distances = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
            if(k > 0):
                distances[inserted_indices] = sys.maxsize

            i = np.argmin(distances)
            
            exemplar_class_set.append(images_indices[i][1].item())
            exemplar_features.append(features[i])
            inserted_indices.append(i)
        
        return exemplar_class_set

    def get_class_images(self,training_set,exemplar_class_set):
        class_images=[]
        for index in exemplar_class_set:
            class_images.append(training_set.__getitem__(index)[0])
        
        return class_images
    
    def compute_centroids(self,net,training_set):
        for exemplar_class_set in self.exemplar_set:
            features = []
            # Extract feature for each exemplar in exemplar_class_set
            class_images = self.get_class_images(training_set,exemplar_class_set)
            with torch.no_grad():
                for img in class_images:
                    img=img.unsqueeze(0)
                    feature = net.feature_extractor(img.cuda())
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm() # Normalize
                    features.append(feature)
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm() # Normalize
                self.exemplar_centroids.append(mu_y)
        
        return self.exemplar_centroids
    
    
    def predict(self,images,net):
        exemplar_means = self.exemplar_centroids
        means = torch.stack(exemplar_means) # (n_classes, feature_size)
        means = torch.stack([means] * len(images)) # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2) # (batch_size, feature_size, n_classes)
        
        with torch.no_grad():
            features = net.feature_extractor(images) # (batch_size, feature_size)
            for i in range(features.size(0)): # Normalize
                features.data[i] = features.data[i] / features.data[i].norm()
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
            clf_loss = clf_criterion(new_output,utils.one_hot_matrix(len(new_output),labels,n_classes*step))
            return clf_loss,clf_loss,clf_loss-clf_loss
        clf_loss = clf_criterion(new_output[:,n_old_classes:],utils.one_hot_matrix(labels,n_classes*step)[:,n_old_classes:])
        dist_loss = dist_criterion(new_output[:,:n_old_classes],sigmoid(old_outputs))
        
        targets = utils.one_hot_matrix(labels,n_classes*step)
        targets[:,:n_old_classes] = sigmoid(old_outputs)
        tot_loss = clf_criterion(new_output,targets)


        return tot_loss,clf_loss/2,dist_loss/2
