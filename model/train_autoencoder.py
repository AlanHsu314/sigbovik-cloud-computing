import os
import torch
from torch import nn, optim
import argparse
from torchvision import transforms
from tqdm import tqdm

from data import make_loaders
from model import Autoencoder, AutoencoderClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--auto_lr', type=float, default=1e-3)
parser.add_argument('--cls_lr', type=float, default=1e-3)
parser.add_argument('--auto_weight_decay', type=float, default=1e-5)
parser.add_argument('--cls_weight_decay', type=float, default=1e-5)
parser.add_argument('--auto_epochs', type=int, default=100)
parser.add_argument('--cls_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--seed', type=int, default=10701)
parser.add_argument('--ckpt_dir', type=str, default=None)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

auto_save_path = os.path.join(args.ckpt_dir, 'auto_ckpt.pth')
cls_save_path = os.path.join(args.ckpt_dir, 'cls_ckpt.pth')

device = torch.device('gpu' if args.gpu else 'cpu')

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
])

train_loader, test_loader = make_loaders(
    args, '../data_collection/data/images',
    '../data_collection/annotations.txt',
    img_transform=img_transform,
)

in_dim = train_loader.dataset[0][0].numel()
hidden_dim = 1
num_autoencoder_layers = 1
num_classes = 4
num_classifier_layers = 420 # DEEEEEEEEEEP NEURAL NETWORK
autoencoder = Autoencoder(in_dim, hidden_dim, num_autoencoder_layers)
print(autoencoder)
autoencoder.to(device)

classifier = AutoencoderClassifier(autoencoder.encoder, hidden_dim, num_classes, num_classifier_layers)
classifier.to(device)

auto_criterion = nn.MSELoss()
auto_optimizer = optim.Adam(autoencoder.parameters(), lr=args.auto_lr, 
    weight_decay=args.auto_weight_decay)

cls_criterion = nn.CrossEntropyLoss()
cls_optimizer = optim.Adam(classifier.parameters(), lr=args.cls_lr, 
    weight_decay=args.cls_weight_decay)

def train_autoencoder(model, loader, criterion, optimizer):
    epoch_loss = 0
    for x, _ in tqdm(loader):
        optimizer.zero_grad()
        x = x.to(device)
        out = model(x)
        loss = criterion(out, x)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach()
    return epoch_loss / len(loader)

def test_autoencoder(model, loader, criterion):
    epoch_loss = 0
    with torch.no_grad():
        for x, _ in tqdm(loader):
            x = x.to(device)
            out = model(x)
            loss = criterion(out, x)
            epoch_loss += loss
    return epoch_loss / len(loader)

def train_classifier(model, loader, criterion, optimizer):
    epoch_loss = 0
    epoch_correct = 0
    for x, y in tqdm(train_loader):
        optimizer.zero_grad()
        x = x.to(device)
        out = model(x)
        pred = torch.argmax(out, dim=-1)
        num_correct = torch.sum(pred == y)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach()
        epoch_correct += num_correct
    return epoch_loss / len(loader), epoch_correct / len(loader.dataset)

def test_classifier(model, loader, criterion):
    epoch_loss = 0
    epoch_correct = 0
    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=-1)
            num_correct = torch.sum(pred == y)
            loss = criterion(out, y)
            epoch_loss += loss
            epoch_correct += num_correct
    return epoch_loss / len(loader), epoch_correct / len(loader.dataset)

print("*** Autoencoder Training ***")
best_test_loss = float('inf')
for epoch in range(args.auto_epochs):
    epoch_str = str(epoch).rjust(2)
    train_loss = train_autoencoder(autoencoder, train_loader, auto_criterion, auto_optimizer)
    print(f"[Epoch {epoch_str}] Train Loss = {train_loss}")
    test_loss = test_autoencoder(autoencoder, train_loader, auto_criterion)
    print(f"[Epoch {epoch_str}] Test Loss = {test_loss}")
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save({
            'autoencoder': autoencoder,
            'optimizer': auto_optimizer
        }, auto_save_path)

state = torch.load(auto_save_path)
autoencoder = state['autoencoder']
classifier.encoder = autoencoder.encoder

best_test_acc = 0
print("*** Classifier Training ***")
for epoch in range(args.cls_epochs):
    epoch_str = str(epoch).rjust(2)
    train_loss, train_acc = train_classifier(classifier, train_loader, cls_criterion, cls_optimizer)
    print(f"[Epoch {epoch_str}] Train Loss = {train_loss}, Train Acc = {train_acc}")
    test_loss, test_acc = test_classifier(classifier, test_loader, cls_criterion)
    print(f"[Epoch {epoch_str}] Test Loss = {test_loss}, Test Acc = {test_acc}")
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save({
            'classifier': classifier,
            'optimizer': cls_optimizer
        }, cls_save_path)
