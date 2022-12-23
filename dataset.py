import torch
import os
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    """
    VOC Dataset class used to load the VOC dataset
    some complexity here due to the label is scaled with respect to image size
    """
    def __init__(self, csv_file, img_dir, label_dir, s=7, b=2, c=20, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.s = s
        self.b = b
        self.c = c

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                boxes.append([class_label, x, y, width, height])
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)
        label_matrix = torch.zeros((self.s, self.s, self.c + 5*self.b))  # the additional 5 nodes won't be used?
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.s * y), int(self.s * x)
            x_cell, y_cell = self.s * x - j, self.s * y - i
            width_cell, height_cell = (width * self.s, height * self.s)

            if label_matrix[i, j, 20] == 0:  # if no obj in i, j
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
