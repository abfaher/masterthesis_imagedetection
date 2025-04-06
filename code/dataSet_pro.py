import os
from torch.utils.data import Dataset
import torch
from torch import Tensor
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
import numpy
import xml.etree.ElementTree as ET  # Importing XML parser
import torchvision.transforms as transforms
 
 
from typing import Tuple, Optional, List, Union, Dict
 
 
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
 
 
        # Check if the number of images in the infrared and visible directories
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
        if self.data_for_SSL: # We take the data from the end of directory to avoid using the same data for training and SSL
            list_dir_infrared = list_dir_infrared[-self.num_images:]
            list_dir_visible = list_dir_visible[-self.num_images:]
        self.infrared_images = [f for f in list_dir_infrared if f.endswith('.jpg')][:self.num_images]
        self.visible_images = [f for f in list_dir_visible if f.endswith('.jpg')][:self.num_images]
        assert self.infrared_images == self.visible_images, 'Different names of infrared and visible images'
 
        # replace the .jpg extension with .xml
        self.annotations = [f.replace('.jpg', '.xml') for f in self.infrared_images]
        assert all([os.path.exists(os.path.join(self.dir_annotations, f)) for f in self.annotations]), 'Some annotations files are missing'
       
        self.transform = transforms.ToTensor()
 
        # Class-to-index mapping
        self.class_to_idx = {}
        self.build_class_mapping()
 
 
 
 
 
    def __getitem__(self, index:int):
 
        infrared, visible, yvisible, yvisible_hist, visible_grad = self.load_image(index)
        boxes, labels = self.get_annotations(index)
 
        # Transform the images to tensors
        infrared = self.transform(infrared)
        visible = self.transform(visible)
        yvisible = self.transform(yvisible)
        yvisible_hist = self.transform(yvisible_hist)
        visible_grad = self.transform(visible_grad)
 
        # Convert boxes and labels to tensors
        boxes = torch.tensor(boxes, dtype=torch.int64)
        labels = torch.tensor(labels, dtype=torch.int64)
 
        if self.data_for_SSL:
            res = (infrared, visible, yvisible, yvisible_hist, visible_grad) # Return only the two image tensors for self-supervised learning
        else:
            #img = torch.cat([infrared, visible, yvisible, yvisible_hist, visible_grad], dim=0) # I do the concatenation of the two modalities here
            img = torch.cat([infrared, visible], dim=0) # I do the concatenation of the two modalities here
 
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
 
            res = (img, target)
 
        return res
 
 
    def __len__(self) -> int:
        return self.num_images
    
 
    def compute_image_gradient(self,img: numpy.ndarray) ->  Image.Image:
        # Sobel gradients in x and y directions
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x direction
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y direction
    
        # Compute gradient magnitude
        grad_magnitude = cv2.magnitude(grad_x, grad_y)
        # Normalize to range 0-255 for visualization
        grad_magnitude = cv2.convertScaleAbs(grad_magnitude)
        grad_magnitude = Image.fromarray(grad_magnitude).convert('L')
        return grad_magnitude
    
 
    def load_image(self, index:int) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
        infrared_image_path = os.path.join(self.dir_infrared, self.infrared_images[index])
        visible_image_path = os.path.join(self.dir_visible, self.visible_images[index])
        infrared = Image.open(infrared_image_path).convert('L')  # Convert to grayscale
        visible = Image.open(visible_image_path)
 
        # convert the visible RGB image to YCbCr and keep only the Y channel
        yvisible, _, _ = visible.convert('YCbCr').split()
 
        # Convert Y channel to histogram equalization
        yvisible_hist = cv2.equalizeHist(numpy.array(yvisible))
        yvisible_hist = Image.fromarray(yvisible_hist)
 
        visible_grad = self.compute_image_gradient(numpy.array(visible))
        
        return infrared, visible, yvisible, yvisible_hist, visible_grad
    
    def build_class_mapping(self) -> None:
        # Note that in the LLVIP dataset, there is only one class ('person'),
        # therefor it don't make sense to do classification with this dataset
 
        # Get the list of unique class names from all the files in annotations directory
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
 
        class_names = list(set(class_names)) # Get unique class names
        class_names.sort() # sort the class names in alphabetical order
 
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        
    
    def get_annotations(self, index: int)-> Tuple[List[List[int]], List[int]]:
 
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
    
 
 
 
    def show_image(self, index:int) -> Tuple[Image.Image, Image.Image]:
 
        infrared, visible, yvisible, yvisible_hist, visible_grad = self.load_image(index)
 
        _, ax = plt.subplots(1, 4, figsize=(30, 10))
        ax[0].imshow(infrared, cmap='gray') # cmap='gray' is used to display the image in grayscale
        ax[0].set_title('Infrared')
        ax[0].axis('off')
        ax[1].imshow(visible)
        ax[1].set_title('Visible')
        ax[1].axis('off')
        ax[2].imshow(yvisible, cmap='gray')
        ax[2].set_title('Y Visible')
        ax[2].axis('off')
        ax[3].imshow(yvisible_hist, cmap='gray')
        ax[3].set_title('Y Visible Histogram Equalization')
        ax[3].axis('off')
        plt.show()
 
        return infrared, visible