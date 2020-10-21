import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_channel=3, out_channel=1):
        super(Discriminator, self).__init__()
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=4, padding=0)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=4, padding=0)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=4, padding=0)
        self.conv5 = nn.Conv2d(512, out_channel, kernel_size=2, stride=4, padding=0)

    def forward(self, x):
        x = self.leaky_relu(x)
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.conv5(x)
        return x

if __name__ == '__main__':
   net = Discriminator(3, 1)
   net = net.cuda()
   from torchsummary import summary
   summary(net, input_size=(3, 256, 256))
