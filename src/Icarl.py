import torch
import numpy as np
from torch import nn
import sys
from torch.nn import functional as F
import random

class Icarl():
    def __init__(self):
        self.exemplar_set=[]
        self.exemplar_centroids=[]
        self.K=2000
    
    def build_exemplars(self,net,images_indices,n_old_classes,n_classes=10):            
        m=int(self.K/(n_old_classes + n_classes))
        print('Build:',m)
        for i in range(n_classes):
            self.exemplar_set.append(self.construct_exemplar_class_set(net,images_indices[i],m))

        return self.exemplar_set

    def build_exemplars_random(self,net,images_indices,n_old_classes,n_classes=10):            
        m=int(self.K/(n_old_classes + n_classes))
        print('Build:',m)
        for i in range(n_classes):
            self.exemplar_set.append(self.construct_exemplar_class_set(net,images_indices[i],m))

        return self.exemplar_set
    
    def reduce_exemplars(self,n_old_classes,n_classes=10):
        m = int(self.K/(n_old_classes+n_classes))
        print('Reduced:',m)
        for i in range(len(self.exemplar_set)):
            self.exemplar_set[i]=self.exemplar_set[i][:m]
        
        return self.exemplar_set

    def random_exemplar_set(self,net,images_indices,m):
        exemplar_class_set = []
        indices = [img_ind[1] for img_ind in images_indices]
        exemplar_class_set = random.sample(indices,m)
    
    def construct_exemplar_class_set(self,net, images_indices, m):
        """
        Args: 
        - net : rete
        - images_indices : lista di tuple contenenti immagine e indice per la classe i-esima classe
        - m : numero di exemplars da selezionare per la i-esima classe

        Return:
        - Lista di indici relativi alle immagini (exemplars) per la i-esima classe
        """
        features = []
        with torch.no_grad():
            for img,_ in images_indices:
                net.train(False)
                img=img.unsqueeze(0) # re-shapa l'immagine in modo tale che la dimensione sia : 1x3x32x32
                feature = net.feature_extractor(img.cuda()).data.cpu().numpy() # estrae la feature map di dimensione 64
                feature = feature / np.linalg.norm(feature) # Normalizza la feature map
                features.append(feature[0]) # lista contenente le features map di ogni immagine

        features = np.array(features)
        class_mean = np.mean(features, axis=0) # centroide per la classe i-esima
        class_mean = class_mean / np.linalg.norm(class_mean) # seconda normalizzazione per il centroide

        inserted_indices =[] # lista contenente gli indici delle immagini già selezionate in modo da non ripescare la stessa immagine
        exemplar_class_set = [] # lista di exemplars selezionati per la i-esima classe
        exemplar_features = [] # lista contenenenti le features per ogni exemplar già selezionato

        for k in range(m):
            S = np.sum(exemplar_features, axis=0) # somma le feature map di ogni exmeplar selezionato
            mu = class_mean # salva il centroide precedentemente computato nella variabile mu
            mu_p = 1.0/(k+1) * (features + S) # sommo le features delle immagini e le features degli exemplars selezionati
            mu_p = mu_p / np.linalg.norm(mu_p) # normalizzo per la seconda volta
            distances = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)) # calcolo la matrice delle distanze
            if(k > 0):
                distances[inserted_indices] = sys.maxsize # bandisco il ri-selezionamento di tutte le immagini già selezionate come exemplars

            i = np.argmin(distances) # seleziono l'immagine che minimizza la distanza 
            
            exemplar_class_set.append(images_indices[i][1].item()) # salvo l'indice dell'immagine scelta come exemplars
            exemplar_features.append(features[i]) # inserisco la feature-map di quell'exemplar nella lista
            inserted_indices.append(i) # inserisco l'indice da bandire alla prossima iterazione
        
        return exemplar_class_set

    def get_class_images(self,training_set,exemplar_class_set):
        class_images=[]
        for index in exemplar_class_set:
            class_images.append(training_set.__getitem__(index)[0])
        
        return class_images
    
    def compute_centroids(self,net,training_set):
        """
        Args: 
        -net : rete
        - training_set : dataset contenente tutte le immagini delle 100 classi
        
        Returns:
        - lista di centroidi già normalizzati due volte
        """
        self.exemplar_centroids=[]
        for exemplar_class_set in self.exemplar_set:
            features = [] #lista contenente tutte le features map degli exemplars selezionati per la i-esima classe
            class_images = self.get_class_images(training_set,exemplar_class_set) # recupero le immagini degli exemplars attraverso gli indici precedentemente selezionati
            with torch.no_grad():
                for img in class_images:
                    net.train(False)
                    img=img.unsqueeze(0) # re-shapa l'immagine in modo tale che la dimensione sia : 1x3x32x32
                    feature = net.feature_extractor(img.cuda()) #estrae la feature map
                    feature = feature.squeeze() # rimuove tutte le dimensioni pari a 1
                    feature.data = feature.data / feature.data.norm() # Normalizza
                    features.append(feature) 
                features = torch.stack(features) 
                mu_y = features.mean(0).squeeze() #calcola il centroide e rimuove tutte le dimensioni pari a 1
                mu_y.data = mu_y.data / mu_y.data.norm() # ri-normalizza
                self.exemplar_centroids.append(mu_y) # inserisce il centroide nella lista di centroidi
        
        return self.exemplar_centroids
    
    
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
