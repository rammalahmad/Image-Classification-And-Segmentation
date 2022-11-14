import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


device = torch.device('cpu')

class Mask:
    def __init__(self):
        """
        # Info
        ---
        This class allows to create the tensor of a segmentation corresonpding to an image
        """
        self.model = torch.load('./script/segmentation_model_2.pth', map_location=device)
        self.model.to(device)
        # model.load_state_dict(torch.load('./script/resnet152_model.h5', map_location=device))
        # model.eval()

    def generate_mask(self, img_path:str)->torch.tensor:
        """
        # Info
        ---
        It creates the tensor of a segmentation corresonpding to an image

        # Params
        ---
        img_path: string, the path of the image that we want to segment

        # Returns
        ---
        Tensor corresponding to the mask
        """
        image = Image.open(img_path)
        input = self.preprocess(image)
        pred_mask = self.model(input)
        pred_mask = pred_mask.detach().squeeze().cpu().numpy()
        return pred_mask


    @staticmethod
    def preprocess(image)-> torch.tensor:
        """
        # Info
        ---
        It embeds and normalizes an image

        # Params
        ---
        Image: the image we want to embed

        # Returns
        ---
        The tensor embedding 
        """
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        input = transform(image)
        return input.to(device).unsqueeze(0)