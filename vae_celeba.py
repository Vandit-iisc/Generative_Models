import torch 
from torch import nn
import torch.nn.functional as F
import numpy
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
import os
import idx2numpy
import matplotlib.pyplot as plt
from torchvision import datasets
import kagglehub
from pathlib import Path
device = torch.device("cuda")
folder_path = Path("Generated_images")
folder_path.mkdir(parents=True,exist_ok=True)


# Download latest version
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

print("Path to dataset files:", path)

train_transforms = transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.CenterCrop(178),
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

data = datasets.ImageFolder(
    root=os.path.join(path,'img_align_celeba'),
    transform=train_transforms,

)

train_loader = DataLoader(
    data,
    shuffle=True,
    batch_size=512,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

TOTAL_EPOCH = 100

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=16,
                               kernel_size=3,
                               stride=1)
        self.conv2 = nn.Conv2d(in_channels=16,
                               out_channels=32,
                               kernel_size=3,
                               stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,
                                     stride=2)
        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3, 
                               stride=1)
        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=3,
                               stride=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2,
                                    stride=2)
        self.conv5 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=3,
                               stride=1)
        self.conv6 = nn.Conv2d(in_channels=256,
                               out_channels=512,
                               kernel_size=3,
                               stride=1)
        self.conv7 = nn.Conv2d(in_channels=512,
                               out_channels=512,
                               kernel_size=3,
                               stride=1)
        self.fc1 = nn.Linear(in_features=2048,
                             out_features=256)
        self.fc2 = nn.Linear(in_features=256,
                             out_features=256)
    def forward(self,x):
        x = self.maxpool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.avgpool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.avgpool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = F.relu(self.conv7(x)).flatten(1)
        x = self.fc2(F.relu(self.fc1(x)))
        d = x.shape
        return x[:,:d[1]//2],x[:,d[1]//2:]
        
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128,512*4*4)
        self.inv_conv1 = nn.ConvTranspose2d(in_channels=512,
                                            out_channels=256,
                                            kernel_size=2,
                                            stride=2)
        self.inv_conv2 = nn.ConvTranspose2d(in_channels=256,
                                            out_channels=128, 
                                            kernel_size=2, 
                                            stride=2)
        self.inv_conv3 = nn.ConvTranspose2d(in_channels=128,
                                            out_channels=64, 
                                            kernel_size=2, 
                                            stride=2)
        self.inv_conv4 = nn.ConvTranspose2d(in_channels=64,
                                            out_channels=3, 
                                            kernel_size=2, 
                                            stride=2)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = x.view(-1,512,4,4)
        x = F.relu(self.inv_conv1(x))
        x = F.relu(self.inv_conv2(x))
        x = F.relu(self.inv_conv3(x))
        x = torch.sigmoid(self.inv_conv4(x))
        return x
    


class VariationAutoEncoder(nn.Module):
    def __init__(self ):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def reparametrize(self,mean,std):
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z
    def forward(self,x):
        diagonal_entries,mean = self.encoder(x)
        std = torch.exp(0.5*diagonal_entries)
        z = self.reparametrize(mean,std)
        op_image_with_channel = self.decoder(z)
        return op_image_with_channel,diagonal_entries,mean
    def only_decoder(self):
        device = next(self.parameters()).device
        with torch.no_grad():
            epsilon = torch.randn(1,128,device=device)
            op_image_with_channel = self.decoder(epsilon)
            return op_image_with_channel
    def generate_image(self,epoch):
        op_image = self.only_decoder().squeeze(0)
        # plt.imshow(op_image.permute(1,2,0).detach().cpu())
        plt.grid("off")
        plt.imsave(f"Generated_images/{epoch}.png", op_image.permute(1,2,0).detach().cpu())
        

    
vae = VariationAutoEncoder()
vae = vae.to(device=device)
optimizer = torch.optim.Adam(params=vae.parameters(), lr=2e-3)
def vae_loss(op_images,input_images,diagonal_entries, mean):
    reconstruction_loss = F.mse_loss(op_images, input_images, reduction="sum")/input_images.shape[0]
    kld_loss = -0.5*torch.sum(1 + diagonal_entries - mean**2 - torch.exp(diagonal_entries))/input_images.shape[0]
    return reconstruction_loss + 0.1 * kld_loss, reconstruction_loss.item(), kld_loss.item()


train_loss = []
Recon_loss = []
KL = []
epochs = list(range(1,TOTAL_EPOCH+1))
def train_vae():
    for epoch in range(TOTAL_EPOCH):
        total_train_loss = 0
        reconstruction_loss = 0
        kld_loss = 0
        for batch_x,_ in train_loader:
            batch_x = batch_x.to(device)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                op_image_with_channel, logvar_diagonal_entries, mean = vae(batch_x)
                loss,recon,kld = vae_loss(op_image_with_channel,batch_x,logvar_diagonal_entries,mean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss+=loss.item()
            reconstruction_loss+=recon
            kld_loss+=kld
        train_loss.append(total_train_loss/len(train_loader))
        Recon_loss.append(reconstruction_loss/len(train_loader))
        KL.append(kld_loss/len(train_loader))
        print(f"\n\nSummary for {epoch+1} epoch: \n Reconstruction Loss: {reconstruction_loss/len(train_loader):.2f}\n KLD Loss: {kld_loss/len(train_loader):.2f}\n Total Loss: {total_train_loss/len(train_loader):.2f}\n")
        vae.generate_image(epoch)

        if (epoch+1)%5==0:
            torch.save(vae.state_dict(), f"Generated_images/vae_{epoch}.pt")
    torch.save(vae.state_dict(), f"final_vae_weight.pt")
train_vae()
plt.figure(figsize=(8,6))
plt.plot(epochs,train_loss, label = 'train loss' )
plt.plot(epochs,Recon_loss, label = 'Reconstruction loss')
plt.plot(epochs,KL, label = 'KL divergence' )
plt.yscale("log")
plt.xlabel("epochs")
plt.ylabel('loss')
plt.title("Train loss vs epochs")
plt.grid(True)
plt.legend()
plt.savefig("Generated_images/loss.jpg")