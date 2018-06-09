from PIL import Image
import os
import os.path
import pandas as pd
import torch.utils.data
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import cv2

def default_image_loader(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, triplets_file_name, transform=None,
                 loader=default_image_loader):

        self.base_path = base_path  
        triplets = []
        data = np.array(pd.read_csv(triplets_file_name))
        self.triplets = data
        self.transform = transform
        self.loader = loader
    
    def __getitem__(self, index):
        img1 = self.loader(os.path.join(self.base_path,self.triplets[index][0]))
        img2 = self.loader(os.path.join(self.base_path,self.triplets[index][1]))
        img3 = self.loader(os.path.join(self.base_path,self.triplets[index][2]))
        img1 = np.reshape(img1, (-1,224,224))
        img2 = np.reshape(img2, (-1,224,224))
        img3 = np.reshape(img3, (-1,224,224))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3
            
    def __len__(self):
        return len(self.triplets) 
