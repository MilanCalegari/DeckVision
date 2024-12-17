from typing import List

import cv2
import numpy as np


class CardSegmentation:
    def __init__(self, min_area=1000):
        self.min_area = min_area

    def calculate_card_locations(self, image_path: str) -> List[np.ndarray]:
        self.img = cv2.imread(image_path)
        if self.img is None:
            print(f"Error: Could not open or read the image file: {image_path}")
            return []

        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cards = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                if len(approx) == 4:
                    cards.append(approx)
        
        # Sort cards from left to right based on their centroid x-coordinate
        card_centroids = []
        for card in cards:
            M = cv2.moments(card)
            cX = int(M["m10"] / M["m00"])
            card_centroids.append((cX, card))
        
        cards = [card for _, card in sorted(card_centroids)]

        return cards
    
    def warp_cards(self, image_path: str, cards: List[np.ndarray]) -> List[np.ndarray]:
        warped_cards = []

        for card_contour in cards:
            pts = card_contour.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")

            s = np.sum(pts, axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")

            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(self.img, M, (maxWidth, maxHeight))
            warped_cards.append(warped)
        return warped_cards
