import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
from torchvision.ops import nms
from pytorch_retinanet.retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from pytorch_retinanet.retinanet.anchors import Anchors
# from pytorch_retinanet.retinanet import losses
from counters.MSR_DRN_pytorch.utils import losses
from counters.MSR_DRN_pytorch.layers._misc import SmoothStepFunction, SpatialNMS, GlobalSumPooling2D
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = nn.Upsample(size=(C4.size(-2), C4.size(-1)), mode='nearest')(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = nn.Upsample(size=(C3.size(-2), C3.size(-1)), mode='nearest')(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        # P6_x = self.P6(C5)

        # P7_x = self.P7_1(P6_x)
        # P7_x = self.P7_2(P7_x)

        return P3_x


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.mid_output1 = nn.Conv2d(feature_size, num_classes, kernel_size=1, padding=0)
        self.mid_act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.mid_output2 = nn.Conv2d(feature_size, num_classes, kernel_size=1, padding=0)
        self.mid_act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.mid_output3 = nn.Conv2d(feature_size, num_classes, kernel_size=1, padding=0)
        self.mid_act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.mid_output4 = nn.Conv2d(feature_size, num_classes, kernel_size=1, padding=0)
        self.mid_act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_classes, kernel_size=1, padding=0)
        self.output_act = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out_middle1 = self.mid_output1(out)
        out_middle1 = self.mid_act1(out_middle1)

        out = self.conv2(out)
        out = self.act2(out)

        out_middle2 = self.mid_output2(out)
        out_middle2 = self.mid_act2(out_middle2)

        out = self.conv3(out)
        out = self.act3(out)

        out_middle3 = self.mid_output3(out)
        out_middle3 = self.mid_act3(out_middle3)

        out = self.conv4(out)
        out = self.act4(out)

        out_middle4 = self.mid_output4(out)
        out_middle4 = self.mid_act4(out_middle4)

        out_final = self.output(out)
        out_final = self.output_act(out_final)

        return out_middle1, out_middle2, out_middle3, out_middle4, out, out_final



class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.classificationModel = ClassificationModel(self.fpn.P3_2.out_channels, num_classes=num_classes)

        self.STF_1 = SmoothStepFunction(threshold=0.4, beta=1)
        self.SNMS = SpatialNMS(kernel_size=3, stride=1, beta=100)
        self.STF_2 = SmoothStepFunction(threshold=0.8, beta=15)
        self.GSP = GlobalSumPooling2D()

        self.final_reg = nn.Linear(257, 1, bias=True)

        # self.anchors = Anchors()
        #
        # self.regressBoxes = BBoxTransform()
        #
        # self.clipBoxes = ClipBoxes()

        self.focal_DRN_loss = losses.focal_DRN
        self.reg_loss_func = nn.L1Loss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, input):

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        P3 = self.fpn([x2, x3, x4])

        # out_middle1, out_middle2, out_middle3, out_middle4, out1
        out_middle1, out_middle2, out_middle3, out_middle4, out, out_final = self.classificationModel(P3)

        GAP_on_final_conv = torch.nn.AvgPool2d(kernel_size=(out.size(-2), out.size(-1)))(out)

        out_final = self.STF_1(out_final)
        out_final = self.SNMS(out_final)
        out_final = self.STF_2(out_final)
        out_final = self.GSP(out_final)

        reg_features = torch.hstack((GAP_on_final_conv.squeeze(3).squeeze(2), out_final))
        final_reg_output = self.final_reg(reg_features)

        return [out_middle1, out_middle2, out_middle3, out_middle4], final_reg_output[0]

        # else:
        #     transformed_anchors = self.regressBoxes(anchors, regression)
        #     transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
        #
        #     finalResult = [[], [], []]
        #
        #     finalScores = torch.Tensor([])
        #     finalAnchorBoxesIndexes = torch.Tensor([]).long()
        #     finalAnchorBoxesCoordinates = torch.Tensor([])
        #
        #     if torch.cuda.is_available():
        #         finalScores = finalScores.cuda()
        #         finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
        #         finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()
        #
        #     for i in range(classification.shape[2]):
        #         scores = torch.squeeze(classification[:, :, i])
        #         scores_over_thresh = (scores > 0.05)
        #         if scores_over_thresh.sum() == 0:
        #             # no boxes to NMS, just continue
        #             continue
        #
        #         scores = scores[scores_over_thresh]
        #         anchorBoxes = torch.squeeze(transformed_anchors)
        #         anchorBoxes = anchorBoxes[scores_over_thresh]
        #         anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
        #
        #         finalResult[0].extend(scores[anchors_nms_idx])
        #         finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
        #         finalResult[2].extend(anchorBoxes[anchors_nms_idx])
        #
        #         finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
        #         finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
        #         if torch.cuda.is_available():
        #             finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()
        #
        #         finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
        #         finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
        #
        #     return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]


def resnet50(args, num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

