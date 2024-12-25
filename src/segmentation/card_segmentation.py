import os
import platform
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from src.utils.config_loader import ConfigLoader
from src.utils.operational_system import OS

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
config = ConfigLoader()


class CardSegmentation:
    def __init__(self) -> None:
        # Load model only once during initialization
        self.model = AutoModelForImageSegmentation.from_pretrained(config.get("segmentation", "model"), trust_remote_code=True)
        torch.set_float32_matmul_precision('high')
        
        # Configure processing device
        if platform == OS.DARWIN:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)
        self.model.eval()

        # Cache configurations
        self.image_size = tuple(config.get("segmentation", "image_size"))
        self.normalize_mean = config.get("segmentation", "normalize_mean")
        self.normalize_std = config.get("segmentation", "normalize_std")

        # Pre-compile transformations
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(self.normalize_mean, self.normalize_std)
        ])

        # Define minimum card area
        self.min_area = 1000

    @torch.no_grad()  # Optimize torch operations by disabling gradients
    def pre_process_image(self, image_path: str):
        self.img = Image.open(image_path)
        input_image = self.transform_image(self.img).unsqueeze(0).to(self.device)
        return input_image

    @torch.no_grad()
    def inference(self, input_image):
        pred = self.model(input_image)[-1].sigmoid().cpu()
        pred = pred[0].squeeze()
        mask = transforms.ToPILImage()(pred)
        mask = mask.resize(self.img.size)
        return mask
    
    def calculate_card_locations(self, img: Image.Image):
        # Convert image only once
        self.img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Optimize image processing
        gray = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Pre-allocate lists for cards
        cards = []
        card_centroids = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                if len(approx) == 4:
                    M = cv2.moments(approx)
                    if M["m00"] != 0:  # Avoid division by zero
                        cX = int(M["m10"] / M["m00"])
                        card_centroids.append((cX, approx))
                        cards.append(approx)

        # Sort cards in a single operation
        return [card for _, card in sorted(card_centroids)]
    
    def warp_cards(self, cards: List[np.ndarray]) -> List[np.ndarray]:
        warped_cards = []

        for card_contour in cards:
            pts = card_contour.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            # Optimize calculations by vectorizing operations
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            # Calculate dimensions in a single operation
            (tl, tr, br, bl) = rect
            width = np.maximum(
                np.sqrt(((br - bl) ** 2).sum()),
                np.sqrt(((tr - tl) ** 2).sum())
            )
            height = np.maximum(
                np.sqrt(((tr - br) ** 2).sum()),
                np.sqrt(((tl - bl) ** 2).sum())
            )

            maxWidth = int(width)
            maxHeight = int(height)

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(self.img_array, M, (maxWidth, maxHeight))
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
            warped_cards.append(warped)

        return warped_cards
    
    def run(self, image_path: str):
        input_image = self.pre_process_image(image_path)
        img = self.img
        
        # First detect how many cards are in the image
        cards = self.calculate_card_locations(img)
        
        # If there is more than one card, perform segmentation before edge detection
        if len(cards) > 1:
            mask = self.inference(input_image)
            img.putalpha(mask)
            cards = self.calculate_card_locations(img)
            
        warped_cards = self.warp_cards(cards)
        return warped_cards