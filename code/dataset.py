import os
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from torch import Tensor
from PIL import Image
import xml.etree.ElementTree as ET  # Importing XML parser
 
 
from typing import Tuple, Optional, List
 
 
class LLVIPDataset(Dataset):
    def __init__(self, dir_all_images: str, train: bool = True, S=7, B=2, C=1, num_images: Optional[int] = None):
        self.train = train
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        self.S = S
        self.B = B
        self.C = C
 
        if self.train:
            self.dir_visible = dir_all_images + '/visible/train'
        else:
            self.dir_visible = dir_all_images + '/visible/test'
 
        self.dir_annotations = dir_all_images + '/Annotations' # This directory contains xml files with the annotations
 
        def count_jpg_files(dir:str)->int:
            return len([f for f in os.listdir(dir) if f.endswith('.jpg')])

        # If num_images is not specified, use all images in the directories
        if num_images is None:
            num_images = count_jpg_files(self.dir_visible)
        self.num_images = num_images

        # get the names of the num_images first images from the directories
        list_dir_visible = os.listdir(self.dir_visible)

        # filter the filnames to keep only the .jpg files
        self.visible_images = [f for f in list_dir_visible if f.endswith('.jpg')][:self.num_images]
        self.annotations = [fname.replace('.jpg','.xml') for fname in self.visible_images]

 
        # Class-to-index mapping
        self.class_to_idx = {}
        self.build_class_mapping()


    def __getitem__(self, index:int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        # loads the images
        visible = self.load_image(index)
        filename = self.visible_images[index]

        # Transform the images to tensors
        visible = self.transform(visible)

        # Initialize the label matrix
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        
        if self.train:
            boxes, labels = self.get_annotations(index)

            for box, label in zip(boxes, labels):
                xmin, ymin, xmax, ymax = box

                # Convert to center coordinates and normalize
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                width = xmax - xmin
                height = ymax - ymin

                # Determine grid cell
                i = min(self.S - 1, int(self.S * y_center))  # row
                j = min(self.S - 1, int(self.S * x_center))  # column

                # Calculate relative position inside that cell
                x_cell = self.S * x_center - j
                y_cell = self.S * y_center - i

                # Scale the width and height to the cell size
                width_cell, height_cell = (
                    width * self.S,
                    height * self.S,
                )

                # Assign values to the label matrix
                if label_matrix[i, j, self.C] == 0:
                    label_matrix[i, j, self.C] = 1  # objectness score
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    label_matrix[i, j, self.C + 1:self.C + 5] = box_coordinates
                    label_matrix[i, j, label] = 1  # one-hot encoding for class

            return visible, label_matrix, filename
        else:
            # For test set, return only the image and filename (No annotations for test)
            return visible, label_matrix, filename

    def __len__(self) -> int:
        return self.num_images
        
    def get_annotations(self, index: int)-> Tuple[List[List[int]], List[int]]:
        # loads the annotations from the xml file
        xml_file = os.path.join(self.dir_annotations, self.annotations[index])
        tree = ET.parse(xml_file)
        root = tree.getroot()
 
        boxes = []
        labels = []

        # Original LLVIP image dimensions
        orig_width = 1280
        orig_height = 1024

        for obj in root.findall('object'):
 
            label = obj.find('name')
            if label is None:
                continue
            label = label.text

            # use the label index corresponding to the class name 
            label_idx = self.class_to_idx[label]
            labels.append(label_idx)
 
 
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
            
            find_xmin = bbox.find('xmin')
            find_ymin = bbox.find('ymin')
            find_xmax = bbox.find('xmax')
            find_ymax = bbox.find('ymax')
            if None in (find_xmin, find_ymin, find_xmax, find_ymax):
                        continue
 
            try:
                xmin = int(find_xmin.text) # type: ignore
                ymin = int(find_ymin.text) # type: ignore
                xmax = int(find_xmax.text) # type: ignore
                ymax = int(find_ymax.text) # type: ignore
            except (TypeError, ValueError):
                continue

            # Normalize bbox coordinates
            xmin /= orig_width
            xmax /= orig_width
            ymin /= orig_height
            ymax /= orig_height
 
            boxes.append([xmin, ymin, xmax, ymax])

        return boxes, labels


    # UTILS FUNCTIONS

    def load_image(self, index:int) -> Image.Image:
        visible_image_path = os.path.join(self.dir_visible, self.visible_images[index])

        visible = Image.open(visible_image_path)

        return visible

    def show_image(self, index:int) -> Image.Image:
        pass

    def build_class_mapping(self) -> None:
        # compute the integer class IDs from the class names because machine learning models don't work with strings

        class_names = []
        for annotation in os.listdir(self.dir_annotations):
            xml_file = os.path.join(self.dir_annotations, annotation)

            # ignore .DS_Store and other junk files
            if not xml_file.endswith('.xml'):
                continue

            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall('object'):
                label = obj.find('name')
                if label is None:
                    continue
                label = label.text
                class_names.append(label)
 
        class_names = list(set(class_names)) # remove duplicates
        class_names.sort() # sort the class names in alphabetical order
 
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    