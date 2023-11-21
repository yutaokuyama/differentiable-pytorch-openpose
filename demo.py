import cv2
import copy

from src import model
from src import util
from src.body import Body

body_estimation = Body('./model/body_pose_model.pth')

test_image = './images/demo.jpg'
oriImg = cv2.imread(test_image)  # B,G,R order

candidate, subset = body_estimation.detect_keypoints(oriImg)
canvas = copy.deepcopy(oriImg)
canvas = util.draw_bodypose(canvas, candidate, subset)

cv2.imwrite("estimated_keypoitns.png",canvas)