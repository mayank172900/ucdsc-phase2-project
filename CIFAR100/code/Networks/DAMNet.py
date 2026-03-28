#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 11:44:59 2022

@author: mlcv
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 22:53:53 2022

@author: mlcv
"""
import math
import torch
# import torchvision.models as models
import torch.nn as nn
import torchvision.models as models
from .model_irse import IR_50
from Networks.iresnet_torch_new import iresnet100

class DAMSequential(nn.Module):
    def __init__(self, input_dim = 128, output_dim=10):
        super(DAMSequential, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.layers = []
        self.layers.append(nn.Linear(input_dim, int(input_dim*0.5)))
        self.layers.append(nn.Sigmoid())
        self.layers.append(nn.Linear(int(input_dim*0.5), output_dim))            
        self.layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        feats, out = self.model(x)
        out = self.layers(feats)
        return out
    
class DAMDebug(nn.Module):
    def __init__(self, input_dim = 128, output_dim=10):
        super(DAMDebug, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.layer1 = nn.Linear(input_dim, int(input_dim*0.5))
        self.layer1_act = nn.Sigmoid()
        self.layer2 = nn.Linear(int(input_dim*0.5), output_dim)  
        self.layer2_act = nn.Sigmoid()
                  
    def forward(self, x):
        feats, out = self.model(x)
        out = self.layer1(feats)
        out = self.layer1_act(out)
        out = self.layer2(out)
        out = self.layer2_act(out)
        return out
  
# x = torch.ones(128,2048)
# dam = DAM(x.shape[1], 256)
# out = dam(x)


# model = models.resnet50(pretrained=True)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 22:53:53 2022

@author: mlcv
"""
import torch
import torchvision.models as models
import torch.nn as nn
   
class DAMSequential(nn.Module):
    def __init__(self, input_dim = 128, output_dim=10):
        super(DAMSequential, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.layers = []
        self.layers.append(nn.Linear(input_dim, int(input_dim*0.5)))
        self.layers.append(nn.Sigmoid())
        self.layers.append(nn.Linear(int(input_dim*0.5), output_dim))            
        self.layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        feats, out = self.model(x)
        out = self.layers(feats)
        return out
    
class DAMDebug(nn.Module):
    def __init__(self, input_dim = 128, output_dim=10):
        super(DAMDebug, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.layer1 = nn.Linear(input_dim, int(input_dim*0.5))
        self.layer1_act = nn.Sigmoid()
        self.layer2 = nn.Linear(int(input_dim*0.5), output_dim)  
        self.layer2_act = nn.Sigmoid()
                  
    def forward(self, x):
        feats, out = self.model(x)
        out = self.layer1(feats)
        out = self.layer1_act(out)
        out = self.layer2(out)
        out = self.layer2_act(out)
        return out
  

class DAMGeneral(nn.Module):
     def __init__(self, input_dim = 128, num_layers=1):
         super(DAMGeneral, self).__init__()
         self.model = iresnet100(pretrained=False)
        #for params in self.model.parameters():
          #   params.requires_grad = False
         input_dim = 512*7*7
         output_dim = 12500
         self.layers = []
         for i in range(num_layers):
             self.layers.append(nn.Linear(input_dim, output_dim))
             #self.layers.append(nn.Sigmoid())

         self.layers = nn.Sequential(*self.layers)
         
     def forward(self, x):
         feats, out = self.model(x)
         dam_out = self.layers(feats)
         return dam_out, dam_out
# model = models.resnet50(pretrained=True)

class DAMGeneralML(nn.Module):
     def __init__(self, input_dim = 128, num_dim=10, num_layers=2):
         super(DAMGeneralML, self).__init__()
         # self.model = models.resnet101(pretrained=True)
         self.model = IR_50(pretrained=True,input_size=[112,112], num_classes = 5088)
         # input_dim = self.model.output_layer.weight.shape[1]
         # self.model = models.resnet50(pretrained=True)
         input_dim=512
         # input_dim = self.model.fc.weight.shape[1]
         self.num_dim = num_dim
         ratio = max(input_dim/num_dim, num_dim/input_dim)
         self.coef = ratio**(1/num_layers)
         self.layers = []
         
         for i in range(num_layers):
             if input_dim*self.coef > self.num_dim:
                 self.layers.append(nn.Linear(input_dim, num_dim))
                 input_dim = self.num_dim
             else:
                 self.layers.append(nn.Linear(input_dim, math.ceil(input_dim*self.coef)))
                 input_dim = math.ceil(input_dim*self.coef)
             self.layers.append(nn.ReLU())
         else:
             self.layers.append(nn.Linear(input_dim, num_dim))

         self.layers = nn.Sequential(*self.layers)
         
     def forward(self, x):
         feats, out = self.model(x)
         dam_out = self.layers(feats)
         return dam_out, dam_out

# x = torch.ones(128,3,112,112)
# dam = DAMGeneral(2048, 256)
# out = dam(x)
