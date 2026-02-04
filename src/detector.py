import cv2
import os

class EarDetector:
    def __init__(self):
        # Paths to Haar cascade files for ears
        # Note: These are often not included in default opencv-python distribution
        # We might need to download them or use a different approach.
        self.left_ear_cascade_path = os.path.join('models', 'haarcascade_mcs_leftear.xml')
        self.right_ear_cascade_path = os.path.join('models', 'haarcascade_mcs_rightear.xml')
        
        self.left_ear_cascade = None
        self.right_ear_cascade = None
        
        if os.path.exists(self.left_ear_cascade_path):
            self.left_ear_cascade = cv2.CascadeClassifier(self.left_ear_cascade_path)
        if os.path.exists(self.right_ear_cascade_path):
            self.right_ear_cascade = cv2.CascadeClassifier(self.right_ear_cascade_path)

    def detect(self, image):
        """
        Detects ears in the image and returns a list of bounding boxes (x, y, w, h).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ears = []
        
        if self.left_ear_cascade:
            left_ears = self.left_ear_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in left_ears:
                ears.append({'box': (x, y, w, h), 'type': 'left'})
                
        if self.right_ear_cascade:
            right_ears = self.right_ear_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in right_ears:
                ears.append({'box': (x, y, w, h), 'type': 'right'})
                
        # If cascades are missing or failed, we might use a dummy or skip
        return ears

    def crop_ear(self, image, box):
        x, y, w, h = box
        return image[y:y+h, x:x+w]
