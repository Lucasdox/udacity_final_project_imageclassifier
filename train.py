import argparse
from torchvision import datasets, transforms, models

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from collections import OrderedDict

from utils import save_checkpoint


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", default="flowers")
    parser.add_argument("--arch", default="densenet121", choices=("vgg16", "densenet121"))
    parser.add_argument("--save_dir", default="checkpoint.pth")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--hidden_units', type=int, default=512)
    
    return parser.parse_args()


def build_model(arch, class_to_idx, hidden_units):
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    if arch == 'vgg16':
        input_node = 25088
    elif arch == 'densenet121':
        input_node = 1024

    classifier = nn.Sequential(OrderedDict([
                                      ('fc1', nn.Linear(input_node, hidden_units)),
                                      ('drop', nn.Dropout(p=0.6)),
                                      ('relu', nn.ReLU()),
                                      ('fc2', nn.Linear(hidden_units, 102)),
                                      ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    model.class_to_idx = class_to_idx
    return model


def train(model, loaders, device, epochs, learning_rate):
    valid_loader = loaders.get("valid_loader")
    test_loader = loaders.get("test_loader")
    train_loader = loaders.get("train_loader")
                                                           
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    print_every = 10

    model.to(device)

    for e in range(epochs):
        running_loss = 0
        print(f"-------Starting Epoch {e+1}--------")
        for ii, (inputs, labels) in enumerate(train_loader):
            inputs, labels = Variable(inputs).to(device), Variable(labels).to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if ii % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for inputs2, labels2 in valid_loader:
                        inputs2, labels2 = Variable(inputs2).to(device), Variable(labels2).to(device)
                        outputs2 = model.forward(inputs2)
                        test_loss += criterion(outputs2, labels2)

                        ps = torch.exp(outputs2)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels2.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                model.train()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(valid_loader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(valid_loader)))
                
    return model, optimizer
                
def main():
    args = get_input_args()
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    save_dir = args.save_dir
    hidden_units = args.hidden_units
    learning_rate = args.learning_rate
    arch = args.arch
  
    #transforms
    data_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    #datasets
    train_data = datasets.ImageFolder(train_dir, transform = data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = test_transforms)

    #loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=20, shuffle=True)
    loaders = dict(
        train_loader = train_loader,
        test_loader = test_loader,
        valid_loader = valid_loader
    )
    
    model = build_model(arch, train_data.class_to_idx, hidden_units)
    device = "cuda" if args.gpu else "cpu"
    epochs = args.epochs
    
    model, optimizer = train(model, loaders, device, epochs, learning_rate)
    
    save_checkpoint(save_dir, arch, model, optimizer, learning_rate)
    
    
    
if __name__ == "__main__":
    main()
