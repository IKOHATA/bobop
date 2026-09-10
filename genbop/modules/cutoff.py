import torch
import numpy as np

class TanhCutoff(torch.nn.Module):
    def __init__(self,Rc,requires_grad = False):
        super().__init__()

        self.Rc = Rc
        self.Bc = torch.nn.Parameter(torch.tensor(1.0),requires_grad=requires_grad)
    
    def cffunc(self,dist):

        x=torch.tanh((1-dist/self.Rc)*self.Bc)/torch.tanh(self.Bc)
        x=torch.pow(x,3)
        return x
    
    def forward(self,dist):

        y=torch.where(dist < self.Rc, self.cffunc(dist), 0)
        
        return y

class PolynomialCutoff(torch.nn.Module):
    def __init__(self,Rc,delta):
        super().__init__()

        self.Rc = Rc
        self.delta = delta
    
    def polynomial(self,dist):

        x = 1-2*(1+(dist-self.Rc)/self.delta)
        y = 1.875*x - 1.25*x**3 + 0.375*x**5

        return (1 + y)/2
    
    def forward(self,dist):
        
        return torch.where( (self.Rc-self.delta < dist) & (dist < self.Rc), self.polynomial(dist), dist < self.Rc)