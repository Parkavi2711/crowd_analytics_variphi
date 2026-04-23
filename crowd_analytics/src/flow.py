import cv2
import numpy as np

class FlowEstimator:
    def __init__(self):
        self.prev_gray = None

    def estimate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.prev_gray = gray
        return np.mean(flow, axis=(0,1))