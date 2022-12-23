from unittest import TestCase
import torch
from model import YoloV1


class TestYolov1(TestCase):
    def test_layers_check(self, s=7, b=2, c=20):
        model = YoloV1(split_size=s, num_boxes=b, num_classes=c)
        x = torch.randn((2, 3, 448, 448))
        model_shape = torch.asarray(model(x).shape).tolist()
        true_shape = [2, 1470]
        self.assertListEqual(model_shape, true_shape)
