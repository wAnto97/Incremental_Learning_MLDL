import torch
from IncrementalLeraningMLDL.src.Exemplars import Exemplars 

class Icarl(Exemplars):
    def __init__(self,K=2000):
       super(Icarl,self).__init__(K)
       

    
