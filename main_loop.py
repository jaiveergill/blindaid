import jetson_utils
import jetson_inference
import os

THRESHOLD = 0.2
timestamp = 0 # in milliseconds

net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=THRESHOLD)
camera = jetson_utils.gstCamera(299, 299, "dev/video1")
display = jetson_utils.glDisplay()

while display.isOpen():
  if 4.9 < round(timestamp, 1) < 5.1:
    # SAVE IMAGE AND RUN MODEL
    pass
  
  
  img, width, height = camera.CaptureRGBA()
  detections = net.Detect(img, width, height)
  display.renderOnce(img, width, height)
  
  seconds_elapsed = 1/net.GetNetworkFPS()
  timestamp += seconds_elapsed
  
