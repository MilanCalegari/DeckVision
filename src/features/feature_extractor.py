import torch
from PIL import Image
from torchvision import models, transforms


class FeatureExtractor:
    def __init__(self) -> None:
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.eval()

        self.transoform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ]
                )
    
    def pre_process_image(self, image_path: str) -> torch.Tensor:
        """"Loads a image from path and convert it to tensor"""
        img = Image.open(image_path).convert("RGB")
        img = self.transoform(img)
        img = img.unsqueeze(0)
        return img

    def extract_features(self, image_path: str):
        img = self.pre_process_image(image_path)

        with torch.no_grad():
            features = self.model(img)
        features = features.flatten().numpy()
        return features

    def extract_features_from_array(self, img_array):
        """Extrai features de um array numpy"""
        img = Image.fromarray(img_array).convert("RGB")
        img = self.transoform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            features = self.model(img)
        features = features.flatten().numpy()
        return features


