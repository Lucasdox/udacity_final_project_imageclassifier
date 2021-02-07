import json

import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt 
from torchvision import transforms
from torch.autograd import Variable
import numpy as np

from utils import load_checkpoint


def get_inputs_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("model_checkpoint_path")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--category_names", default="cat_to_name.json")
    
    return parser.parse_args()


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil = Image.open(image)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = transform(pil)
    
    
    return image.numpy()


def predict(image_path, category_names, model, topk):
    cat_to_name = None
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    model.eval()
    with torch.no_grad():
        model.to('cuda')
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = Variable(image.float()).to('cuda')

        outputs = model.forward(image)

        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(topk, dim=1)
        
        inv_class_to_idx = {v: k for k, v in model.class_to_idx.items()}
        top_classes_list = top_class[0].tolist()
        top_p_list = top_p[0].tolist()
        
        top_classes = [inv_class_to_idx[x] for x in top_classes_list]
        top_labels = [cat_to_name[x] for x in top_classes]
        
    
    return top_labels, top_classes


def main():
    args = get_inputs_args()
    image_path = args.image_path
    model_checkpoint_path = args.model_checkpoint_path
    topk = args.topk
    category_names = args.category_names
    device = "cuda" if args.gpu else "cpu"
    
    model = load_checkpoint(model_checkpoint_path)
    top_labels, top_classes = predict(image_path, category_names, model, topk)
    print(f"labels: {top_labels}\nclasses: {top_classes}")
    

if __name__ == "__main__":
    main()