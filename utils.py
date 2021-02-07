import torch


def save_checkpoint(path, arch, model, optimizer, learning_rate):
    model.cpu()
    
    checkpoint = {'arch': arch, 
                  'model': model,
                  'epochs': 10,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    
    if arch == 'resnet101':
        checkpoint['fc'] = model.fc
    else:
        checkpoint['classifier'] = model.classifier
    file = str()
    if path != None:
        file = path + '/' + arch + '_checkpoint.pth'
    else:
        file = arch + '_checkpoint.pth'

    torch.save(checkpoint, file)
    print(f"Model {arch} has been saved into path {file}")
    
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    if checkpoint.get('classifier'):
        model.classifier = checkpoint['classifier']
    elif checkpoint.get('fc'):
        model.fc = checkpoint['fc']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model