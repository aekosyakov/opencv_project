# import the necessary packages
import numpy as np
import argparse
import cv2
from matplotlib import pyplot as plt
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

# define the list of boundaries
boundaries = [
              ([17, 15, 100], [50, 56, 200]),
              ([86, 31, 4], [220, 88, 50]),
              ([25, 146, 190], [62, 174, 250]),
              ([103, 86, 65], [145, 133, 128])
              ]

# loop over the boundaries
for (lower, upper) in boundaries:
      # create NumPy arrays from the boundaries
      lower = np.array(lower, dtype = "uint8")
      upper = np.array(upper, dtype = "uint8")
      
      im = image
      im[im >= 128]= 255
      im[im < 128] = 0
      cv2.imwrite('out.jpg', im)


      # find the colors within the specified boundaries and apply
      # the mask
      mask = cv2.inRange(image, lower, upper)
      output = cv2.bitwise_and(image, image, mask = mask)
#      
#      lower_black = np.array([0,0,0], dtype = "uint16")
#      upper_black = np.array([70,70,70], dtype = "uint16")
#      black_mask = cv2.inRange(image, lower_black, upper_black)
#      image[np.where((image == [0,0,0]).all(axis = 2))] = [0,255,255]
#      black_mask[np.where(black_mask == [0])] = [255]

      # skeletonize the image
#      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#      skeleton = imutils.skeletonize(gray, size=(3, 3))
      #cv2.imshow("Skeleton", skeleton)

#      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#      edgeMap = imutils.auto_canny(gray)

      # show the images
      plt.imshow(mask)
      plt.show()


def alphaCutoutNumpy(im):
    data = numpy.array(im)
    data_T = data.T
    r, g, b, a = data_T
    blackAlphaAreas = (a == 0)
    data_T[0][blackAlphaAreas] = 0
    data_T[1][blackAlphaAreas] = 0
    data_T[2][blackAlphaAreas] = 0
    #data_T[3][blackAlphaAreas] = 255
    plt.imshow(Image.fromarray(data[:,:,:3]))
    plt.show()
# return Image.fromarray(data[:,:,:3])


