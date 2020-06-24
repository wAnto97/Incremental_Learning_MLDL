from torchvision.datasets import CIFAR100
from torch.utils.data import Subset
from torchvision.transforms import ToPILImage
import random 
from sklearn.model_selection import train_test_split


# splitting the 100 classes in [num_groups] groups
# and the indexes of the images belonging to those classes as well
def get_n_splits(dataset, n_groups,random_state=41):
  
  available_labels = list(range(100))
  n_classes_group = int(100 / n_groups)
  labels = []
  indexes = [[] for i in range(n_groups)]
  random.seed(random_state)

  for index in range(n_groups):
    labels_sample = random.sample(available_labels,n_classes_group)
    labels.append(labels_sample)
    available_labels = list(set(available_labels) - set(labels_sample))

  for index in range(len(dataset)):
    label = dataset.__getitem__(index)[1]
    for i in range(n_groups):
      if labels[i].__contains__(label):
        indexes[i].append(index)
        break

  return indexes,labels



# IncrementalCIFAR class stores the CIFAR100 dataset and some info helpful for the 
# incremental learning process: the splitting of the groups and of the indexes
class MyCIFAR100():

  def __init__(self, root, n_groups = 10, train=True, transform=None, target_transform=None, download=False, random_state=653):
        self.dataset = CIFAR100(root, train=train, transform = None, target_transform=None, download=download)
        self.n_groups = n_groups
        self.n_classes_group = int(100 / n_groups)
        self.random_state = random_state
        self.indexes_split,self.labels_split = get_n_splits(self.dataset, n_groups,random_state = self.random_state)
        self.sorted_labels = []
        self.transform = transform
        self.target_transform = lambda target : self.sorted_labels.index(target)

        for l in self.labels_split:
            self.sorted_labels += l

        
  def __getitem__(self,index):
    
    if(self.transform):
        image = self.transform(self.dataset[index][0])
    if(self.target_transform):
        target = self.target_transform(self.dataset[index][1])

    return image,target,index

  # get the subset of the dataset relative to the [group_index] group

  def get_groups_joint(self, n_groups):
    indexes = []  
    for group in range(n_groups):
      indexes += self.indexes_split[group]
    return Subset(self, indexes)

  def get_group(self, group):
    indexes = self.indexes_split[group-1]
    return Subset(self, indexes)
  
  def get_train_val_group(self,group):
      indexes = self.indexes_split[group-1]

      train_indexes,val_indexes = train_test_split(indexes,test_size=0.1,\
      stratify = [self.dataset.__getitem__(i)[1] for i in indexes],random_state=41)
 
      train_dataset = Subset(self, train_indexes)
      val_dataset = Subset(self, val_indexes)

      return train_dataset,val_dataset

  def get_train_val_joint(self,n_groups):
    indexes = []
    for group in range(n_groups):
      indexes += self.indexes_split[group]

    train_indexes,val_indexes = train_test_split(indexes,test_size=0.1,\
    stratify = [self.dataset.__getitem__(i)[1] for i in indexes],random_state=41)
 
    train_dataset = Subset(self, train_indexes)
    val_dataset = Subset(self, val_indexes)

    return train_dataset,val_dataset

  def get_single_train_joint_validation(self,group):
    indexes = []
    train_indexes = []

    for gr in range(group):
      indexes += self.indexes_split[gr]

    train_indexes_tmp,val_indexes = train_test_split(indexes,test_size=0.1,\
    stratify = [self.dataset.__getitem__(i)[1] for i in indexes],random_state=41)

    for index in train_indexes_tmp:
      label = self.target_transform(self.dataset.__getitem__(index)[1])
      if(label >= (group-1)*self.n_classes_group):
          train_indexes.append(index)

    train_dataset = Subset(self, train_indexes)
    val_dataset = Subset(self, val_indexes)

    return train_dataset,val_dataset     

  def get_train_exemplars(self,group,exemplars_matrix):
    indexes = []
    train_indexes = []
    exemplar_indexes = []

    for gr in range(group):
      indexes += self.indexes_split[gr]

    for index in indexes:
      label = self.target_transform(self.dataset.__getitem__(index)[1])
      if(label >= (group-1)*self.n_classes_group):
          train_indexes.append(index)

    for i in range(len(exemplars_matrix)): # Flat map to retrieve all exemplars indexes
      exemplar_indexes += exemplars_matrix[i]

    train_dataset = Subset(self, train_indexes + exemplar_indexes)
    
    return train_dataset


  def get_train_val_exemplars(self,group,exemplars_matrix):
    indexes = []
    train_indexes = []
    exemplar_indexes = []

    for gr in range(group):
      indexes += self.indexes_split[gr]
      
    train_indexes_tmp,val_indexes = train_test_split(indexes,test_size=0.1,\
    stratify = [self.dataset.__getitem__(i)[1] for i in indexes],random_state=41)

    for index in train_indexes_tmp:
      label = self.target_transform(self.dataset.__getitem__(index)[1])
      if(label >= (group-1)*self.n_classes_group):
          train_indexes.append(index)


    for i in range(len(exemplars_matrix)): # Flat map to retrieve all exemplars indexes
      exemplar_indexes += exemplars_matrix[i]

    train_dataset = Subset(self, train_indexes + exemplar_indexes)
    val_dataset = Subset(self,val_indexes)
    
    return train_dataset,val_dataset

  def get_train_rebalancing(self,group,exemplars_matrix):
    indexes = []
    train_indexes = []
    exemplar_indexes = []

    
    indexes = self.indexes_split[group-1]

    for i in range(len(exemplars_matrix)): # Flat map to retrieve all exemplars indexes
      exemplar_indexes += exemplars_matrix[i]

    train_indexes_new,val_indexes_new = train_test_split(indexes,train_size=int(len(exemplar_indexes)/(group-1)),\
    stratify = [self.dataset.__getitem__(i)[1] for i in indexes],random_state=41)

    train_dataset = Subset(self, exemplar_indexes + train_indexes_new)
    
    return train_dataset