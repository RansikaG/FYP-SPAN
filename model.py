import torch
import torch.nn as nn
import torchvision.transforms as T
from resnet import resnet34, resnet50, CNN1, CNN2, resnet50_SecondHalf
import math
from torchsummary import summary


class Generator_Block(nn.Module):
    def __init__(self):
        super(Generator_Block, self).__init__()
        nc, nz, ngf = 1, 256, 64
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # state size. nz x 24 x 24
            nn.ConvTranspose2d(nz, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 48 x 48
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 96 x 96
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 192 x 192
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        x = self.main(input)
        return x


class PartAtt_Generator(nn.Module):
    def __init__(self):
        super(PartAtt_Generator, self).__init__()
        self.extractor = resnet34()
        self.generator_front = Generator_Block()
        self.generator_rear = Generator_Block()
        self.generator_side = Generator_Block()

    def forward(self, x):
        x = self.extractor(x, 3)
        front = self.generator_front(x)
        rear = self.generator_rear(x)
        side = self.generator_side(x)
        return torch.cat([front, rear, side], 1)


class Foreground_Generator(nn.Module):
    def __init__(self):
        super(Foreground_Generator, self).__init__()
        self.extractor = resnet34()
        nc, nz, ngf = 1, 256, 64
        self.generator = nn.Sequential(
            # input is Z, going into a convolution
            # state size. nz x 24 x 24
            nn.ConvTranspose2d(nz, ngf * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 27 x 27
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 30 x 30
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 60 x 60
        )

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.extractor(x, 3)
        x = self.generator(x)
        x = x.view(-1, 60, 60)
        return x


class Second_Stage_Extractor(nn.Module):
    def __init__(self):
        super(Second_Stage_Extractor, self).__init__()
        self.stage1_extractor = CNN1().eval()  # we don't train the stage 1 extractor
        self.stage2_extractor_global = CNN2(num_features=1024)
        self.stage2_extractor_front = CNN2()
        self.stage2_extractor_rear = CNN2()
        self.stage2_extractor_side = CNN2()

    def forward(self, image, front_mask, rear_mask, side_mask):
        # masks should be 24x24
        # front_mask, rear_mask, side_mask = image_masks

        global_stage_1 = self.stage1_extractor(image)
        front_image = torch.mul(global_stage_1, front_mask)
        rear_image = torch.mul(global_stage_1, rear_mask)
        side_image = torch.mul(global_stage_1, side_mask)

        global_features = self.stage2_extractor_global(global_stage_1)
        front_features = self.stage2_extractor_front(front_image)
        rear_features = self.stage2_extractor_rear(rear_image)
        side_features = self.stage2_extractor_side(side_image)
        return torch.cat((global_features, front_features, rear_features, side_features), dim=1)


class Second_Stage_Resnet50_Features(nn.Module):
    def __init__(self):
        super(Second_Stage_Resnet50_Features, self).__init__()
        self.stage1_extractor = CNN1().eval()  # we don't train the stage 1 extractor
        self.stage2_features_global = resnet50_SecondHalf().eval()
        self.stage2_features_front = resnet50_SecondHalf().eval()
        self.stage2_features_rear = resnet50_SecondHalf().eval()
        self.stage2_features_side = resnet50_SecondHalf().eval()

    def forward(self, image, front_mask, rear_mask, side_mask):
        # masks should be 24x24
        # front_mask, rear_mask, side_mask = image_masks

        global_stage_1 = self.stage1_extractor(image)
        front_image = torch.mul(global_stage_1, front_mask)
        rear_image = torch.mul(global_stage_1, rear_mask)
        side_image = torch.mul(global_stage_1, side_mask)

        global_resnet_features = self.stage2_features_global(global_stage_1)
        front_resnet_features = self.stage2_features_front(front_image)
        rear_resnet_features = self.stage2_features_rear(rear_image)
        side_resnet_features = self.stage2_features_side(side_image)
        return torch.cat((global_resnet_features, front_resnet_features, rear_resnet_features, side_resnet_features),
                         dim=1)


class FC_Features(nn.Module):
    def __init__(self, input_size=2048 * 12 * 12, global_size=1024, other_view_size=512):
        super(FC_Features, self).__init__()
        self.global_size = global_size
        self.view_size = other_view_size
        # input_size = 18432
        input_size = 2048
        self.input_size = input_size
        self.global_fc = nn.Linear(input_size, global_size)
        self.front_fc = nn.Linear(input_size, other_view_size)
        self.rear_fc = nn.Linear(input_size, other_view_size)
        self.side_fc = nn.Linear(input_size, other_view_size)

    def forward(self, features):
        # features is one dimensional vector [global, front, rear, side]
        global_resnet_features = features[:, :self.input_size]
        front_resnet_features = features[:, self.input_size: 2 * self.input_size]
        rear_resnet_features = features[:, 2 * self.input_size: 3 * self.input_size]
        side_resnet_features = features[:, 3 * self.input_size:]

        global_features = self.global_fc(global_resnet_features)
        front_features = self.front_fc(front_resnet_features)
        rear_features = self.rear_fc(rear_resnet_features)
        side_features = self.side_fc(side_resnet_features)

        return torch.cat((global_features, front_features, rear_features, side_features), dim=1)


class BoatIDClassifier(nn.Module):
    def __init__(self, input_size=2560, num_of_classes=100):
        super(BoatIDClassifier, self).__init__()
        self.input_size = input_size
        self.num_classes = num_of_classes
        self.batchNorm = nn.BatchNorm1d(self.input_size)
        self.fc = nn.Linear(self.input_size, self.num_classes)

    def forward(self, features):
        # features should be of size 2560 = 1024 + 512 + 512 + 512
        # out = self.batchNorm(features)
        out = self.fc(features)
        # No softmax since softmax is applied in the cross entropy loss
        return out


if __name__ == '__main__':
    model =  nn.Sequential(resnet50(), nn.MaxPool2d(4, stride=3), nn.Flatten())
    # model = nn.Sequential(resnet50(), nn.Flatten())
    print(summary(model, (3, 192, 192), device='cpu'))
