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

# For visdom
from visdom import Visdom
import matplotlib.pyplot as plt
    
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

class VisdomLinePlotter(object):
    """Plots line graphs to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom(port=7000)
        self.env = env_name
        self.plots = {}
    
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), 
                                                 Y=np.array([y,y]), 
                                                 env=self.env, opts=dict(
                                                 legend=[split_name],
                                                 title=title_name,
                                                 xlabel='Iterations',
                                                 ylabel=var_name
            ))
            print(self.plots)
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), 
                          env=self.env, win=self.plots[var_name], 
                          name=split_name, update = 'append')
            print(x, y)

def main():
    # Training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 10
    NUM_CLASSES = 3
    LR = 1e-3
    USE_VISDOM = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    dirs = { # TODO: make train-val-test the parent dir
        'train': './datasets/stories/train/',
        'val': './datasets/stories/val/',
    }
    
    # Experiment tracking
    test_name = "2_with_best"
    notes = "confusion_matrix"
    if USE_VISDOM:
        global plotter
        plotter = VisdomLinePlotter(env_name='StoryNet')
    
    # Model setup
    ckpt_name = './checkpoints/1_from_scratch_weight_decay_bestf1_lr0.001000.pth' # None
    model = gen_model(ckpt=ckpt_name, device=device).to(device)
#     summary(model, (1, 200, 200))
#     return
    
    # Training setup
    transform = T.Compose([ # TODO: add flips, shear, small rotations
        T.Grayscale(),
        T.ToTensor(),
        T.Normalize([0], [255]),
    ])
    
    train_data = ImageFolder(root=dirs["train"], transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    val_data = ImageFolder(root=dirs["val"], transform=transform)
    val_loader = DataLoader(val_data, batch_size=50, shuffle=True)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.001)
    
    # Training loop
    best_f1 = 0
    for it in range(NUM_EPOCHS):
        #-- TRAIN --#
        losses = []
        for imgs, labels in train_loader:
            # Load data
            imgs = imgs.to(device)
            # labels best kept as integer class, https://stackoverflow.com/a/62456801
            labels = labels.long().to(device) 
            
            # Predict
            preds = model(imgs)
            print(preds)
            print(labels)
            loss = loss_fn(preds, labels)
            losses.append(loss.detach().cpu().numpy())
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        #-- VALIDATE --#
        pred_history = []
        label_history = []
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs)

            pred_history = np.append(pred_history, np.argmax(preds.detach().cpu().numpy(), axis=1))
            label_history = np.append(label_history, labels.detach().cpu().numpy())
        
        
        f1 = f1_score(label_history, pred_history, average='macro')
        confusion = confusion_matrix(label_history, pred_history)
        
        #-- REPORT --#
        print("It %d, loss %.5f, val F1 %.5f" % (it, losses[-1], f1))
        print("Confusion", confusion)
        if USE_VISDOM:
            plotter.plot('loss', 'train', 'NLL Cross Entropy Loss', it, losses[-1])
            plotter.plot('f1', 'val', 'Macro F1 Score', it, f1)

        if f1 > best_f1: 
            ckpt_name = './checkpoints/%s_%s_bestf1_lr%f.pth' % (test_name, notes, LR)
            torch.save(model.state_dict(), ckpt_name)
            best_f1 = f1
            print("Saved weights as best F1")
            
    ckpt_name = './checkpoints/%s_%s_latest_lr%f.pth' % (test_name, notes, LR)
    torch.save(model.state_dict(), ckpt_name)

if __name__ == "__main__":
    main()