import os
import random
import pickle
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.cifar_dataset = datasets.CIFAR10(root=root_dir, train=train, download=True, transform=transform)
        self.classes = self.cifar_dataset.classes
        self.transform = transform
        self.cls_img, self.cls_lbs = [[] for _ in range(10)], [[] for _ in range(10)]
        for i in range(len(self.cifar_dataset.targets)):
            for j in range(len(self.classes)):
                if self.cifar_dataset.targets[i] == j:
                    self.cls_img[j].append(self.cifar_dataset.data[i])
                    self.cls_lbs[j].append(self.cifar_dataset.targets[i])
        
        #print(len(self.cls_img[3]), self.cls_lbs[3])

    def __getitem__(self, index):
        class_index = self.cifar_dataset.targets[index]
        #print(index)
        class_images = []
        class_labels = []

        class_images = self.cls_img[class_index]
        class_labels = self.cls_lbs[class_index]

        random_samples = random.sample(list(zip(class_images, class_labels)), 2)

        random_images, random_labels = zip(*random_samples)
        # img1 = torch.tensor(random_images[0])
        # img2 = torch.tensor(random_images[1])
        img1 = self.cifar_dataset.data[index]
        #img1 = random_images[0].astype(np.float32) / 127.5 - 1
        img1 = img1.astype(np.float32) / 127.5 - 1
        img2 = random_images[1].astype(np.float32) / 127.5 - 1

        return (img1, img2), class_index, self.classes[class_index]


    def __len__(self):
        # Return the number of classes in CIFAR10
        return len(self.cifar_dataset.targets)


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    cifar_dataset = CIFAR10Dataset(root_dir='/home/lhy/Projects/EDM_Diffusion/data', train=True, transform=transform)
    cifar_dataset_test = CIFAR10Dataset(root_dir='/home/lhy/Projects/EDM_Diffusion/data', train=False, transform=transform)

    train_data_loader= DataLoader(cifar_dataset, batch_size=512, shuffle=True)
    test_data_loader= DataLoader(cifar_dataset_test, batch_size=512, shuffle=False)
    # for (image1, image2), lab, class_name in train_data_loader:
    #     print(image1.shape, type(image2), lab, class_name)


    # paths, classes, classes_name= list_image_files_and_class_recursively(train_image_path)
    # transform = transforms.Compose([transforms.ToTensor()])
    # train_data = ImageDataset(image_path=train_image_path, image_size=32, paths=paths, classes=classes, classes_name=classes_name,transform=transform)
    # train_data_loader= DataLoader(train_data, batch_size=1, shuffle=True)
    # for (image1,image2), classes, classes_name in train_data_loader:
    #     tensor_to_image = transforms.ToPILImage()
    #     image = tensor_to_image(image1.view(-1,32, 32))
    #     image.save("image1.jpg")

    model = torchvision.models.resnet50(pretrained=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
    #optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mode  = model.to(device)

    model.train()
    for epoch in range(10):
        step = 0
        for (image1, image2), lab, class_name in tqdm(train_data_loader, desc=f"Epoch", unit="batch"):
            inputs,labels = image1, lab
            inputs = inputs.permute(0, 3, 1, 2)
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 20 == 19:
                print(f"Epoch:{epoch+1}, loss: {loss.item()}")
            step += 1
        print('epoch{} loss:{:.4f}'.format(epoch+1,loss.item()))

    torch.save(model,'cifar10_densenet161.pt')
    print('cifar10_densenet161.pt saved')


    model = torch.load('cifar10_densenet161.pt')

    model.eval()

    correct,total = 0,0
    for (image1, image2), lab, class_name in test_data_loader:
        inputs,labels = image2,lab
        inputs = inputs.permute(0, 3, 1, 2)

        inputs,labels = inputs.to(device),labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data,1)
        total =total+labels.size(0)
        correct = correct +(predicted == labels).sum().item()

    print('test acc:{:.4f}%'.format(100.0*correct/total))
