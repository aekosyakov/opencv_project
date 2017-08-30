import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# load the image
cv_image = cv2.imread(args["image"])

# Load the face detection cascade file.
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load our Martian as foreground image with alpha transparency.
# The -1 reads the alpha transparency of our image otherwise known as the face hole.
foreground_image = cv2.imread('astromask.png', -1)

# Create foreground mask from alpha transparency.
foreground_mask = foreground_image[:, :, 3]

# Create inverted background mask.
background_mask = cv2.bitwise_not(foreground_mask)

# Convert foreground image to BGR.
foreground_image = foreground_image[:, :, 0:3]

# Declare foreground size.
foreground_size = 630
foreground_ratio = float(foreground_size)

# Declare background size and padding.
background_size = 1100

padding_top = int(((background_size - foreground_size) / 3) * 2)
padding_bottom = int(background_size - padding_top)
padding_left = int((background_size - foreground_size) / 2)
padding_right = int((background_size - foreground_size) / 2)

# Find that face.
faces = face_cascade.detectMultiScale(
                                    cv_image,
                                    scaleFactor=1.1,
                                    minNeighbors=3,
                                    minSize=(30, 30),
                                    flags=0
                                    )





# Iterate over each face found - roi: region of interest
for (x1, y1, w, h) in faces:
  
      # Extract image of face.
      x2 = x1 + w
      y2 = y1 + h
      
      face_roi = cv_image[y1:y2, x1:x2]
      
      # Resize image of face.
      ratio = foreground_ratio / face_roi.shape[1]
      dimension = (foreground_size, int(face_roi.shape[0] * ratio))
      face = cv2.resize(face_roi, dimension, interpolation = cv2.INTER_AREA)
      
      # Add padding to background image
      background_image = cv2.copyMakeBorder(face, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_CONSTANT)
      
      # Region of interest for Martian from background proportional to martian size.
      background_src = background_image[0:background_size, 0:background_size]
      
      # roi_bg contains the original image only where the martian is not
      # in the region that is the size of the Martian.
      roi_bg = cv2.bitwise_and(background_src, background_src, mask=background_mask)
      
      # roi_fg contains the image of the Martian only where Martian is
      roi_fg = cv2.bitwise_and(foreground_image, foreground_image, mask=foreground_mask)
      
      # Join the roi_bg and roi_fg.
      dst = cv2.add(roi_bg, roi_fg)
      
      # Write the final image back to file path, overwriting original image.
      cv2.imwrite('face_crop1.jpg', dst)
