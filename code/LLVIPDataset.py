import os
from torch.utils.data import Dataset
import torch
from torch import Tensor
from PIL import Image
import torchvision.transforms as transforms
import numpy
import xml.etree.ElementTree as ET  # Importing XML parser
import torchvision.transforms as transforms
 
 
from typing import Tuple, Optional, List
 
 
class LLVIPDataset(Dataset):
    def __init__(self, dir_all_images:str, train:bool=True, num_images: Optional[int]=None, data_for_SSL:bool=False):
        self.train = train
 
        if self.train:
            # The directories where the training images (.jpg) are stored
            self.dir_infrared = dir_all_images + '/infrared/train'
            self.dir_visible = dir_all_images + '/visible/train'
        else:
            # The directories where the test images (.jpg) are stored
            self.dir_infrared = dir_all_images + '/infrared/test'
            self.dir_visible = dir_all_images + '/visible/test'
 
        self.dir_annotations = dir_all_images + '/Annotations' # This directory contains xml files with the annotations
 
        self.data_for_SSL = data_for_SSL # If True, the dataset is used for self-supervised learning (unsupervised data without labels) and we take the data from the end of directory to avoid using the same data for training and SSL

        # check if the number of infrared and visible images are the same
        def count_jpg_files(dir:str)->int:
            return len([f for f in os.listdir(dir) if f.endswith('.jpg')])
        assert count_jpg_files(self.dir_infrared) > 0, f'No .jpg files in {self.dir_infrared}'
        assert count_jpg_files(self.dir_infrared) == count_jpg_files(self.dir_visible), 'Different number of infrared and visible images'

        # If num_images is not specified, use all images in the directories
        if num_images is None:
            num_images = count_jpg_files(self.dir_infrared)
        self.num_images = num_images

        # get the names of the num_images first images from the directories
        list_dir_infrared = os.listdir(self.dir_infrared)
        list_dir_visible = os.listdir(self.dir_visible)

        # filter the filnames to keep only the .jpg files
        self.infrared_images = [f for f in list_dir_infrared if f.endswith('.jpg')][:self.num_images]
        self.visible_images = [f for f in list_dir_visible if f.endswith('.jpg')][:self.num_images]
        assert self.infrared_images == self.visible_images, 'Different names of infrared and visible images'

        self.transform = transforms.ToTensor()
 
        # Class-to-index mapping
        self.class_to_idx = {}
        self.build_class_mapping()


    def __getitem__(self, index:int) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        # loads the images
        visible, infrared = self.load_image(index)
        boxes, labels = self.get_annotations(index)

        # Transform the images to tensors
        infrared = self.transform(infrared)
        visible = self.transform(visible)

        # Convert boxes and labels to tensors
        boxes = torch.tensor(boxes, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)

        image = visible # we only use the visible image for now
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        res = (image, target)

        return res


    def __len__(self) -> int:
        return self.num_images
        
    def get_annotations(self, index: int)-> Tuple[List[List[int]], List[int]]:
        # loads the annotations from the xml file
        xml_file = os.path.join(self.dir_annotations, self.annotations[index])
        tree = ET.parse(xml_file)
        root = tree.getroot()
 
        boxes = []
        labels = []

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
 
            boxes.append([xmin, ymin, xmax, ymax])
 
        labels = [label + 1 for label in labels] # Add one because the background class is 0
 
        return boxes, labels


    # UTILS FUNCTIONS

    def load_image(self, index:int) -> Image.Image:
        infrared_image_path = os.path.join(self.dir_infrared, self.infrared_images[index])
        visible_image_path = os.path.join(self.dir_visible, self.visible_images[index])

        infrared = Image.open(infrared_image_path).convert('L')  # Convert to grayscale (but we won't use it for now)
        visible = Image.open(visible_image_path)

        return visible, infrared

    def show_image(self, index:int) -> Image.Image:
        pass

    def build_class_mapping(self) -> None:
        # compute the integer class IDs from the class names becuase machine learning models don't work with strings

        class_names = []
        for annotation in os.listdir(self.dir_annotations):
            xml_file = os.path.join(self.dir_annotations, annotation)
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
    