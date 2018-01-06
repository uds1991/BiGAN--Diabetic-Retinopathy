import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
#X_dim = 128*128*3
z_dim = 100
img = 128


class Generator(nn.Module):
    
     def __init__(self):
            

            super(Generator, self).__init__()

            self.main = nn.Sequential(
                nn.ConvTranspose2d(z_dim, img * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(img * 16),
                nn.LeakyReLU(0.2,True),
                nn.ConvTranspose2d(img*16, img * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img * 8),
                nn.LeakyReLU(0.2,True),
                nn.ConvTranspose2d(img * 8, img * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img * 4),
                nn.LeakyReLU(0.2,True),
                nn.ConvTranspose2d(img * 4, img*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img*2),
                nn.LeakyReLU(0.2,True),
                nn.ConvTranspose2d(img * 2, img, 4, 2, 1, bias=False),
                nn.BatchNorm2d(img),
                nn.LeakyReLU(0.2,True),
                nn.ConvTranspose2d( img , 3, 4, 2, 1, bias=False),
                nn.Tanh()        
               )
        
       
     def forward(self, input):
        
        return self.main( input )

class Encoder(nn.Module):
    
    def __init__(self):#### E_z is the encoder output size. Here it is 100 dimension feature map(Noise vector).
        

        super(Encoder, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 8, stride=2, padding=2),     ## 128 +4 -8/2 + 1 = 63
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      ## 63-3/2 +1 = 31
            nn.Conv2d(64, 192, kernel_size=5, padding=2),                 ## 31 - 5 +4 /1 + 1 = 31
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(kernel_size=3, stride=2),                       ## 31-3/2 + 1 = 15
            nn.Conv2d(192, 384, kernel_size=3, padding=1),                ## 15 - 3 +2/1 + 1 = 15
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2,True),
            nn.MaxPool2d(kernel_size=3, stride=2),                        ## 15 - 3/2 +1 = 7
            nn.Conv2d(256, 128, kernel_size = 3, padding = 1),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024), ## Size calculated output= (input - kernel)/stride + 1  
            nn.LeakyReLU(0.2,True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2,True),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512,100)
        )
       



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    
class Discriminator(nn.Module):
    
    
    def __init__(self):#### E_z is the encoder output size. Here it is 100 dimension feature map(Noise vector).
        

        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 8, stride=2, padding=2),  ## 128 +4 -8/2 + 1 = 63
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True),
            nn.Dropout2d(p = 0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),                    ## 63-3/2 +1 = 31
            nn.Conv2d(64, 192, kernel_size=5, padding=2),     ## 31 - 5 +4 /1 + 1 = 31
            nn.BatchNorm2d(192), 
            nn.LeakyReLU(0.2,True),
            nn.Dropout2d(p = 0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),                    ## 31-3/2 + 1 = 15
            nn.Conv2d(192, 384, kernel_size=3, padding=1),            ## 15 - 3 +2/1 + 1 = 15
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2,True),
            nn.Dropout2d(p = 0.5),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True),
            nn.Dropout2d(p = 0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2,True),
            nn.Dropout2d(p = 0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),                     ## 15 - 3/2 +1 = 7
            nn.Conv2d(256,128, kernel_size = 7),                     ## ouput will be 128 * 1 * 1
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout2d(p = 0.5),
        )
        
   
        self.flatten = nn.Sequential(
              nn.Linear(128,128, bias = True),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Dropout(p = 0.5), 
                                                        ## output of this will be a vector of size 128
              nn.Linear(128,128, bias = True),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Dropout(p = 0.5)                       ## output of this will be a vector of size 128
        )
    
        self.inference_z = nn.Sequential(
              # input dim: z_dim x 1 x 1
              nn.Linear(100, 128, bias=True),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Dropout2d(0.5),
              # state dim: 512 x 1 x 1
              nn.Linear(128, 128, bias=True),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Dropout2d(p=0.5)
                                                  # output dim: 128 x 1 x 1                
                                                ## Output of nn.inference_z will be vector of 128 obtained using a noise vector
        )
        
        self.inference_joint = nn.Sequential(
 
              nn.Linear(256, 256, bias=True),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Dropout2d(p=0.5),
           
              nn.Linear(256, 256, bias=True),
              nn.LeakyReLU(0.2, inplace=True),
              nn.Dropout2d(p=0.5),
             
              nn.Linear(256, 1, bias=True),
              nn.Sigmoid()
            
                                                            
        )
    
    def forward(self, x, z):
            x = self.features(x)
            x = x.view(x.size(0),-1)    ## Output_x size  = (128,1)
            x = self.flatten(x)                       ## x size  = (128,1)
            z = z.view(z.size(0),-1)
            z = self.inference_z(z)                  ## output_z size = (128,1)
            X = torch.cat((x,z),1)                   ## X size = (256,1)
            output = self.inference_joint(X)                ## output size = 1
            return output    


