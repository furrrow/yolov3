import torch
import torch.nn as nn

v1_config = [
    # Tuple: (kernel_size, num_filters, stride, padding)
    # List: [tuples*, repeats]
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]

# tuple: (out_channels, kernel_size, stride)
# everything in yolov3 uses the same padding
# List: ["B" = res blocks, repeat]
v3_config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class YoloV1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YoloV1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fc = self._create_fc(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fc(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                ks, num_filters, s, p = x
                layers += [
                    CNNBlock(in_channels, num_filters, kernel_size=ks, stride=s, padding=p)
                ]
                in_channels = num_filters
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(x) == list:
                conv1, conv2, num_repeats = x
                ks1, num_filters1, s1, p1 = conv1
                ks2, num_filters2, s2, p2 = conv1
                for _ in range(num_repeats):
                    layers += [CNNBlock(in_channels, num_filters1, kernel_size=ks1, stride=s1, padding=p1)]
                    layers += [CNNBlock(num_filters1, num_filters2, kernel_size=ks2, stride=s2, padding=p2)]
                    in_channels = num_filters2
        return nn.Sequential(*layers)

    def _create_fc(self, split_size, num_boxes, num_classes):
        s, b, c = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * s * s, 512),  # original paper had 4096
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(512, s * s * (c + b * 5))  # (s, s, 30) where c + b*5 = 30
            )

