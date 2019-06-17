from ipdb import set_trace as bp
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import imshow
import torch.nn.functional as F
import torch
from tqdm import tqdm
from multitask_model import HNet
# from constant import epochs
import h5py

total_epoch = 30
batch_size = 100
device = 'gpu' ;
net = HNet().to(device)

class etl(torch.utils.data.Dataset):

    def __init__(self, split, transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File('data.h5', 'r', driver='core')

        if self.split == 'training':
            self.train_datas = self.data['training_pixel']
            self.train_labels_age = self.data['training_label_age']
            self.train_labels_gender = self.data['training_label_gender']
            self.train_datas = np.asarray(self.train_datas)
        else:
            self.test_datas = self.data['testing_pixel']
            self.test_labels_age = self.data['testing_label_age']
            self.test_labels_gender = self.data['testing_label_gender']
            self.test_datas = np.asarray(self.test_datas)

    def __getitem__(self, index):

        if self.split == 'training':
            img, target1, target2 = self.train_datas[index], self.train_labels_age[index], self.train_labels_gender[index]
        else:
            img, target1, target2 = self.test_datas[index], self.test_labels_age[index], self.test_labels_gender[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target1, target2

    def __len__(self):
        if self.split == 'training':
            return len(self.train_datas)
        else:
            return len(self.test_datas)


print(">>>>>>>> Loading data <<<<<<<<<<")

train_transforms = transforms.Compose([transforms.ToTensor()
                                      ])
test_transforms = transforms.Compose([transforms.ToTensor()
                                      ])
test_data = etl("testing", test_transforms)
train_data = etl("training", train_transforms)
print (len(test_data))
best_acc = 0
real_best_acc = 0

link_temp = "./checkpoint/*.pt"
real_link_temp = "./checkpoint/real_*.pt"
link_temp = link_temp.replace("*", "Hnet")
real_link_temp = real_link_temp.replace("*", 'HNet')
print(link_temp)
print(real_link_temp)

if os.path.isfile(link_temp) == False:
    state = {
        'net': net.state_dict(),
        'acc': 0,
        'epoch': 0,
    }
    start_epoch = state['epoch']
    best_acc = state['acc']

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, link_temp)
else:
    assert os.path.isdir('checkpoint'), 'Error!'
    checkpoint = torch.load(link_temp)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# if os.path.isfile(real_link_temp) == True:
#     real_checkpoint = torch.load(real_link_temp)
#     real_best_acc = checkpoint['acc']
#
# real_test_data = datasets.ImageFolder("data/test", transform=test_transforms)


train_load = torch.utils.data.DataLoader(dataset=train_data,
                                         batch_size=batch_size,
                                         shuffle=True)

test_load = torch.utils.data.DataLoader(dataset=test_data,
                                        batch_size=batch_size,
                                        shuffle=False)

# real_test_load = torch.utils.data.DataLoader(dataset=real_test_data,
#                                         batch_size=batch_size,
#                                         shuffle=False)

def lr_decay(epoch):

    # 0.1 for epoch  from 0 to 150
    # 0.01 for epoch from 150,250
    # 0.001 for epoch [250,350)

    if epoch < 1:
        return 0.3

    if epoch < 3:
        return 0.2

    if epoch < 150:
        return 0.1

    if epoch < 250:
        return 0.001

    if epoch < 350:
        return 0.0001

    return 0.00001

def train(epoch):

    _lr = lr_decay(epoch)
    net.train()

    optimizer = optim.SGD(net.parameters(), lr=_lr,
                          momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=_lr, weight_decay=5e-4)

    # criterion = nn.BCELoss()

    train_loss = 0
    correct1 = 0
    correct2 = 0
    total1 = 0
    total2 = 0
    now = time.time()

    print('''
           Starting training:
               Epochs: {}
               Batch size: {}
               Learning rate: {}
               Training size: {}
               Test size: {}
               Checkpoints: {}
               CUDA: {}
           '''.format(epoch, batch_size,  _lr, len(train_data),
                      len(train_load),link_temp, device))





    # targer1 = age ; targer2 = gender
    for batch_idx, (inputs, targets_age, targets_gender) in enumerate(train_load):

        inputs, targets_age, targets_gender = inputs.to(device), targets_age.to(device), targets_gender.to(device)

        optimizer.zero_grad()
        output1, output2 = net(inputs)
        # _, predicted1 = output1.max(1)
        # _, predicted2 = output2.max(1)
        # correct2 += predicted2.eq(targets_gender).sum().item()
        loss1 = F.cross_entropy(output1, targets_age)
        loss2 = F.cross_entropy(output2, targets_gender)
        loss = loss1 + loss2;
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted1 = output1.max(1)
        total1 += targets_age.size(0)
        correct1 += predicted1.eq(targets_age).sum().item()

        _, predicted2 = output2.max(1)
        total2 += targets_gender.size(0)
        correct2 += predicted2.eq(targets_gender).sum().item()

        if (batch_idx + 1) % 20 == 0:
            loss_now = (train_loss / (batch_idx + 1))
            time_now = time.time() - now
            total_data = len(train_data)
            done_data = batch_idx * batch_size
            print("Data: {} / {} Loss:{:.7f}  Time:{:.2f} s".format(done_data,
                                                                    total_data, loss_now, time_now))
            now = time.time()

def test(epoch):

    global best_acc, link_temp
    net.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets_age, targets_gender) in enumerate(tqdm(test_load)):
            inputs, targets_age, targets_gender = inputs.to(device), targets_age.to(device), targets_gender.to(device)

            output1, output2 = net(inputs)
            # loss1 = F.cross_entropy(output1, targets_age)
            # loss2 = F.cross_entropy(output2, targets_gender)
            # loss = loss1 + loss2;
            # test_loss += loss.item()
            _, predicted1 = output1.max(1)
            _, predicted2 = output2.max(1)
            total += targets_age.size(0)
            correct1 += predicted1.eq(targets_age).sum().item()
            correct2 += predicted2.eq(targets_gender).sum().item()
            # print("Batch: {}/{} Loss:{0.4f}".format(batch_idx, len(testloader), (test_loss / (batch_idx + 1))))
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    # print(correct)
    # print(total)
    acc1 = (100 * correct1) / total
    acc2 = (100 * correct2) / total
    acc = (acc1 + acc2)/2
    print("Accuracy_age: {:.3f}%".format(acc1))
    print("Accuracy_gender: {:.3f}%".format(acc2))
    if acc > best_acc:
        print('Saving..')
        print('Best accuracy: {:.3f}%'.format(acc))
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        torch.save(state, link_temp)
        best_acc = acc

def main():

    print("Start epoch: ", start_epoch)
    print("Total_epoch: ", total_epoch)
    print("Best now: ", best_acc)
    print("Batch size: ", batch_size)
    # ultra-test
    for idx in range(start_epoch, total_epoch):
        train(idx)
        test(idx)

if __name__ == "__main__":
    # pass
    # print(os.path.isfile('./checkpoint/resnet18.pt'))
    # print(">>> done <<<")
    print(">>> hey <<<")
    main()
    print(">>> done <<<")