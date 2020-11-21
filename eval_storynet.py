# For training
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, confusion_matrix

from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries

def gen_model(ckpt=None, device='cpu'):
    """
    Loads a fully convolutional neural network class that uses 3 layers to represent
    distinguishing features between different stories. Optional initialization to saved weights.
    """
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=1),
        nn.AvgPool2d(66,66),
        nn.Flatten(),
        nn.Linear(64, 3)
    )
    
    if ckpt:
        print("Loading model weights from checkpoint ", ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device(device)))
    else:
        print("Loading randomly initialized weights")
    return model

def main():
    # Training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 1000
    NUM_CLASSES = 3
    LR = 1e-3
    USE_VISDOM = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def batch_predict(images):
        # Model setup
        ckpt_name = './checkpoints/1_from_scratch_weight_decay_bestf1_lr0.001000.pth'
        model = gen_model(ckpt=ckpt_name, device=device).to(device)
        model.eval()

        # Training setup
        transform = T.Compose([ # TODO: add flips, shear, small rotations
            T.ToTensor(),
            T.Normalize([0], [255]),
        ])

        # LIME specific setup
        print(images)
        
        batch = torch.stack(tuple(transform(i) for i in images), dim=0)
        batch = batch.to(device)
        
        preds = model(batch)
        probs = F.softmax(preds, dim=1)
        return probs.detach().cpu().numpy()

    dirs = { # TODO: make train-val-test the parent dir
        'train': './datasets/stories/train/',
        'val': './datasets/stories/val/',
    }
    
#     val_data = ImageFolder(root=dirs["val"], transform=transform)
#     val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    img_path = 'datasets/stories/val/abraham_isaac/72.36-orphan_o2.jpg'
    images = [Image.open(img_path)]
    
#     print(batch_predict(images))
    
    pil_transform = T.Compose([T.Grayscale(),])
#     print(pil_transform(images[0]))
#     print(np.array(pil_transform(images[0])).shape)
    
#     return
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(pil_transform(images[0])), 
                                         batch_predict, # classification function
                                         top_labels=3, 
                                         hide_color=0, 
                                         num_samples=8) # number of images that will be sent to classification function
    

if __name__ == "__main__":
    main()