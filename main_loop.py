import jetson_utils
import jetson_inference
import os

THRESHOLD = 0.2

net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=THRESHOLD)
camera = jetson_utils.gstCamera(299, 299, "dev/video1")
display = jetson_utils.glDisplay()

while display.isOpen():
  img, width, height = camera.CaptureRGBA()
  detections = net.Detect(img, width, height)
  display.renderOnce(img, width, height)
  
