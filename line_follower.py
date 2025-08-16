import cv2
import numpy as np

LOWER_RED1 = np.array([0, 100, 100])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([160, 100, 100])
UPPER_RED2 = np.array([179, 255, 255])

class LineFollower:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get_red_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
        mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
        mask = cv2.bitwise_or(mask1, mask2)
        return mask

    def get_offset(self, frame):
        # 只取下方1/3区域分析
        roi_y1 = int(self.height * 0.6)
        roi = frame[roi_y1:, :].copy()
        mask = self.get_red_mask(roi)

        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            offset = cx - self.width // 2
            return offset, True
        else:
            return None, False