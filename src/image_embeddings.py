from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np

def get_inception_activations(images, model, device, batch_size=32):
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    activations = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch = [preprocess(Image.open(img.filepath).convert('RGB')) for img in batch]
            batch = torch.stack(batch)
            batch = batch.to(device)
            pred = model(batch)[0]
            activations.append(pred.cpu().numpy())
    
    activations = np.concatenate(activations, axis=0)
    return activations

def calculate_fid(real_activations, syn_activations):
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = syn_activations.mean(axis=0), np.cov(syn_activations, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_inception_score(activations, splits=10):
    scores = []
    for i in range(splits):
        part = activations[(i * activations.shape[0] // splits):((i + 1) * activations.shape[0] // splits), :]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean(np.sum(part * np.log(part / py), axis=1))))
    return np.mean(scores), np.std(scores)
