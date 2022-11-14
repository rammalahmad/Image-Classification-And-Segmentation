import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image


class Model:
    def __init__(self):
        """
        # Info
        ---
        This class allows to classify an image as containing silo or not
        """
        device = torch.device('cpu')
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        model.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        model.to(device)
        model.load_state_dict(torch.load('./script/resnet152_model.h5', map_location=device))
        model.eval()
        
        self.model = model
        # self.path = r"C:\Users\User\Desktop\Ahmad\Hackathon\ai_ready\images"

    def labelise(self, img_path:str)->int:
        """
        # Info
        ---
        It labelizes the image

        # Params
        ---
        img_path: string, the path of the image that we want to classify

        # Returns
        ---
        Image label as 1: contains silo and 0: does not contain silo
        """
        image = Image.open(img_path)
        input = self.preprocess(image)
        output = self.model(input)
        return torch.argmax(output).item()

    @staticmethod
    def preprocess(image)->torch.tensor:
        """
        # Info
        ---
        It embeds and normalizes an image

        # Params
        ---
        Image: the image we want to classify

        # Returns
        ---
        The tensor embedding 
        """
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        input = transform(image)
        return input.reshape((1,3,224,224))