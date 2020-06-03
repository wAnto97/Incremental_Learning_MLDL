import torch
import sys
from torch.nn import functional as F
from IncrementalLeraningMLDL.src.Exemplars import Exemplars 

class Icarl(Exemplars):
    def __init__(self,K=2000):
       super(Icarl,self).__init__(K)
       
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
    
