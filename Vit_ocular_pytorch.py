import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision.io import read_image
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sys import stdout
import os, time
from torch.optim.lr_scheduler import ExponentialLR
from Confuse import ConfuseMatrix


class img_normalize():

    def __init__(self):
        pass


    def __call__(self, img):
        if img.dtype == torch.uint8:
            img = img.to(torch.float32)
            img /= 255
        elif (img.dtype == np.uint8) or (img.dtype == np.uint16):
            img = img.astype('float32')
            img /= 255
        return img
    


class OcularDataset(Dataset):

    def __init__(self, datadir, csv, transforms = None):
        self.csv = csv
        self.datadir = datadir
        self.transform = transforms


    def __len__(self):
        return len(self.csv)
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.datadir,
                                self.csv.iloc[idx, -1])
        image = read_image(img_name)
        label = self.csv.iloc[idx, -2]

        if self.transform:
            image = self.transform(image)
        return image, label


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.patchify = nn.Unfold(kernel_size = patch_size, stride = patch_size)
        self.mlp = nn.Linear(in_channels * patch_size ** 2, embed_dim)
        
      # self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.patchify(x)
        x = x.transpose(-1, -2)
        return F.relu(self.mlp(x))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-head attention
        x1 = self.norm1(x)
        x2 = x + self.attn(x1, x1, x1)[0]
        # MLP
        x3 = self.norm2(x2)
        x3 = x2 + self.mlp(x3)
        return x3


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, num_heads=12, mlp_dim=3072, num_layers=12, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder Blocks
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        batch_size = x.shape[0]

        # Append class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Pass through Transformer layers
        for layer in self.encoder:
            x = layer(x)

        x = self.norm(x)
        x = torch.mean(x, dim = 1)
        # print(x.shape)
        # cls_token_final = x[:, 0]

        # Classification head
        x = F.sigmoid(self.head(x))
        return x


# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # First conv layer
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Second conv layer
#         self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Fully connected layer
#         self.fc2 = nn.Linear(128, 8)          # Output layer (10 classes for 10 digits)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))  # First conv layer with ReLU activation
#         x = F.max_pool2d(x, 2)     # Max pooling
#         x = F.relu(self.conv2(x))  # Second conv layer with ReLU activation
#         x = F.max_pool2d(x, 2)     # Max pooling
#         x = x.view(-1, 64 * 56 * 56) # Flatten the tensor for fully connected layer
#         x = F.relu(self.fc1(x))    # Fully connected layer with ReLU
#         x = F.sigmoid(self.fc2(x))            # Output layer
#         return x

def progressbar(max, iter, start_time, dash_len = 50, starter_txt = ''):
    f = starter_txt + '[' + ''.join(['-']*dash_len)+']'
    stdout.write('\r')
    f = f.replace('-', '#', int((iter+1)/max*dash_len))
    f += f'{(iter+1)/max * 100:.2f} ### {time.time()-start_time:.1f} elapsed.'
    stdout.write(f)
    stdout.flush()


def train(model, device, train_dataloader, optimizer, criterion, threshold = .5):
    model.train()
    dsize = len(train_dataloader)
    cm = None
    outer_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Zero the gradients
        output = model(images)   # Forward pass
        loss = criterion(output, labels)  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights
        outer_loss += loss.cpu().item()
        predict = (output >= threshold).float()
        ccm = ConfuseMatrix(labels.cpu().numpy(), predict.cpu().numpy())
        progressbar(dsize, i, start_time)
    cm += ccm
    return outer_loss, cm


@torch.no_grad()
def test(model, device, test_dataloader, criterion, threshold = .5):
    model.eval()
    dsize = len(test_dataloader)
    cm = None
    outer_loss = 0
    start_time = time.time()
    for i, (images, labels) in enumerate(test_dataloader):
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        outer_loss += criterion(output, labels).cpu().item()
        predict = (output >= threshold).float()
        ccm = ConfuseMatrix(labels.cpu().numpy(), predict.cpu().numpy())
        progressbar(dsize, i, start_time, 10)
    cm += ccm
    return outer_loss, cm


def save_checkpoint(model, epoch, optimizer, loss, filename):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,  # Save the current loss
    }
    torch.save(state, filename)


def load_checkpoint(filename="model_checkpoint.pth", device = 'cuda'):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename, weights_only = False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(3)
    # device = 'cuda'

    ## data loading
    datadir = 'ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/Training Images'
    csvdir = 'ocular-disease-recognition-odir5k/full_df.csv'
    csv = pd.read_csv(csvdir)
    temp = ([np.array(eval(csv.iloc[i, -2]), dtype = 'float32') for i in range(len(csv))])
    csv.target = temp
    batchsize = 100
    transform = transforms.Compose([transforms.Resize((224, 224)), img_normalize()])
    dataset = OcularDataset(datadir, csv, transform)
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [.7, .2, .1])
    num_workers = 6
    train_dataloader = DataLoader(train_dataset, batch_size = batchsize, shuffle=False, num_workers= num_workers, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batchsize, shuffle=False, num_workers= 2, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size = batchsize, shuffle=False, num_workers= 2, pin_memory=True)

    ## hyper parameters
    epoch = 100
    img_size = 224
    patch_size = 16
    embed_size = 320
    att_head = 8 
    mlp_size = embed_size
    dropout = 0
    tblock = 4
    if not os.path.exists('./images'):
        os.mkdir('./images')
    if not os.path.exists('./models'):
        os.mkdir('./models')
    
    sf = f'./models/ViT_{epoch}_{img_size}_{patch_size}_{embed_size}_{att_head}_{mlp_size}_{tblock}_{dropout}.pth'
    sf_best = f'./models/ViT_{epoch}_{img_size}_{patch_size}_{embed_size}_{att_head}_{mlp_size}_{tblock}_{dropout}_best.pth'

    psf = f'./images/ViT_{epoch}_{img_size}_{patch_size}_{embed_size}_{att_head}_{mlp_size}_{tblock}_{dropout}.jpg'
    psfr = f'./images/ViT_{epoch}_{img_size}_{patch_size}_{embed_size}_{att_head}_{mlp_size}_{tblock}_{dropout}_result.jpg'
    ## model
    model = VisionTransformer(img_size, patch_size, 3, 8, embed_size, att_head, mlp_size, tblock, dropout*.1)
    # model = SimpleCNN()
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    ##


    history = {'loss':[], 'accuracy':[], 'val_loss':[], 'val_accuracy':[]}
    min_val_loss = 1e100
    for i in range(epoch):
        print(f'epoch: {i+1:d}')
        loss, cm = train(model, device, train_dataloader, optimizer, criterion)
        print('\nval:')
        val_loss, val_cm = test(model, device, valid_dataloader, criterion)
        # scheduler.step()
        history['loss'].append(loss)
        history['accuracy'].append(cm.acc())
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_cm.acc())
        print(f'  train: accuracy = {cm.acc():.2f}, loss = {loss:.4f}', end = '')
        print(f'---val: accuracy = {val_cm.acc():.2f}, loss = {val_loss:.4f}')
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_checkpoint(model, i, optimizer, loss, sf_best)
    
    
    plt.figure(figsize = (16, 8))
    plt.subplot(1,2,1)
    plt.plot(np.arange(epoch), history['loss'], c = 'b', label = 'train')
    plt.plot(np.arange(epoch), history['val_loss'], '*--r', label = 'val')
    plt.title('loss')

    plt.subplot(1,2,2)
    plt.plot(np.arange(epoch), history['accuracy'], c = 'b', label = 'train')
    plt.plot(np.arange(epoch), history['val_accuracy'], '--r', label = 'val')
    plt.title('accuracy')

    plt.legend()

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,  # Save the current loss
    }
    torch.save(state, sf)
    
    print(f'\n#------------------------{epoch} epoch------------------------#')
    test_dataloader = DataLoader(test_dataset, batch_size= 100, shuffle=True)
    result = test(model, device, test_dataloader, criterion)
    a = f"loss = {result[0]:.4f}, accuracy = {result[1].acc():.2f}%"
    print(a)
    

    print(f'#------------------------best------------------------#')
    load_checkpoint(filename = sf_best)
    result = test(model, device, test_dataloader, criterion)
    b = f"loss = {result[0]:.4f}, accuracy = {result[1].acc():.2f}%"
    print(b)
    plt.savefig(psf)


    plt.figure(figsize=(16, 16))
    j = 1
    with torch.no_grad():
        model.eval()
        correct = 0
        outer_loss = 0
        for i, (images, labels) in enumerate(test_dataloader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            outer_loss += criterion(output, labels).cpu().item()
            predict = output.round()
            correct += predict.eq(labels.view_as(predict)).sum().item()
            j = 0
            for k in range(batchsize):
                j = j+1
                plt.subplot(4, 5, j)
                plt.imshow(images[k].cpu().permute(1, 2, 0).numpy())
                plt.axis('off')
                plt.title(f'true: {np.where(labels[k].cpu().numpy() == 1)[0].tolist()}, predict = {np.where(predict[k].cpu().numpy() == 1)[0].tolist()}')
                if j >= 20:
                    break
                # plt.title(f'True: {lbls_id[1][lbls_id[0] == i]}, predict: {predicts_id[1][predicts_id[0] == i]}')
            if j > 10:
                break
    fig = plt.gcf()
    fig.supxlabel(a + '\n' + b, position = (0, 1e6), horizontalalignment = 'left')
    plt.savefig(psfr)