import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple
# import geotorch

# Support: ['IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False)

        nn.init.xavier_uniform_(self.fc1.weight.data)

        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False), BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth))
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]

    return blocks


class Backbone(Module):
    def __init__(self, input_size, num_layers,include_top=False,orthonormal_const=False,num_classes=285, mode='ir'):
        super(Backbone, self).__init__()
        assert input_size[0] in [32, 112, 224], "input_size should be [32, 32], [112, 112] or [224, 224]"
        assert num_layers in [50, 100, 152], "num_layers should be 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.num_classes = num_classes
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.feat_dim = 512 #512
        self.include_top = include_top
        if input_size[0] == 112:
            if(orthonormal_const):
                self.output_layer = Sequential(BatchNorm2d(512),
                                               Dropout(),
                                               Flatten(),
                                               # Linear(512 * 7 * 7, self.feat_dim,bias=False),
                                               BatchNorm1d(self.feat_dim))
            else:
                self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 7 * 7, 512*7*2))
                                           # Linear(512 * 7 * 7, self.feat_dim),
                                           # PReLU(self.feat_dim))
                                           # BatchNorm1d(self.feat_dim))
        elif input_size[0] == 32:
            self.output_layer = Sequential(BatchNorm2d(512),
                                           Dropout(),
                                           Flatten(),
                                           Linear(512 * 2 * 2, self.feat_dim),
                                           # PReLU(self.feat_dim))
                                            BatchNorm1d(self.feat_dim))
        else:
            print('something goes wrong with input size')
        if(include_top):
            self.fc = nn.Linear(self.feat_dim, num_classes)
        print('model created.')
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)
        #self.register_buffer('centers', torch.zeros(num_classes, 512))
        self._initialize_weights()
        # if(orthonormal_const):
        #     geotorch.orthogonal(self.output_layer[-2], "weight")
        if self.num_classes > self.feat_dim:
            self.relu = ReLU(inplace=True)
            self.agmmt_layer =  nn.Linear(self.feat_dim, num_classes-1)
    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        # if self.num_classes > self.feat_dim:
        #     x = self.relu(x)
        #     # x = F.avg_pool2d(x.view(x.shape[0],x.shape[1],1,1), 1).squeeze()
        #     return self.agmmt_layer(x), x
        # else: 
        return x, x
        # if(self.include_top):
        #     y = self.fc(x)
        #     return y
        # else:
        #     return x
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


def IR_50(pretrained,input_size=[112,112],include_top=False,orthonormal_const=False,num_classes=285):
    """Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, include_top, orthonormal_const, num_classes)
    if(pretrained):
        checkpoint_ = torch.load('/run/media/mlcv/DATA_SSD/Hasan/NirvanaFace/Networks/backbone_ir50_ms1m_epoch63.pth')
        # checkpoint_.pop('output_layer.3.weight')
        # checkpoint_.pop('output_layer.3.bias')
        # checkpoint_.pop('output_layer.4.weight')
        # checkpoint_.pop('output_layer.4.bias')
        # checkpoint_.pop('output_layer.4.running_mean')
        # checkpoint_.pop('output_layer.4.running_var')
        model.load_state_dict(checkpoint_,strict=False) 
        print('Checkpoint loaded.')
    return model


def IR_101(pretrained, input_size=[112,112], include_top=False, orthonormal_const=False, num_classes=285):
    """Constructs a ir-101 model.
    """
    # model = Backbone(input_size, 100, 'ir')
    model = Backbone(input_size, 100, include_top, orthonormal_const, num_classes)
    if(pretrained):
            checkpoint_ = torch.load('/run/media/mlcv/DATA_SSD/Hasan/NirvanaFace/Networks/backbone_ir50_ms1m_epoch63.pth')
            # checkpoint_.pop('output_layer.3.weight')
            # checkpoint_.pop('output_layer.3.bias')
            # checkpoint_.pop('output_layer.4.weight')
            # checkpoint_.pop('output_layer.4.bias')
            # checkpoint_.pop('output_layer.4.running_mean')
            # checkpoint_.pop('output_layer.4.running_var')
            model.load_state_dict(checkpoint_,strict=False) 
            print('Checkpoint loaded.')
    return model


def IR_152(input_size):
    """Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model


def IR_SE_50(input_size):
    """Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model


def IR_SE_101(input_size):
    """Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model


def IR_SE_152(input_size):
    """Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model



# test_model = IR_50(True,input_size=[32,32])
# test_input = torch.zeros(4,3,32,32)
