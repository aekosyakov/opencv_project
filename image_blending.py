import cv2

# Read the images
foreground = cv2.imread("portrait.jpg")
background = cv2.imread("clouds.jpg")
alpha = cv2.imread("puppets_alpha.png")

# Convert uint8 to float
foreground = foreground.astype(float)
background = background.astype(float)

# Normalize the alpha mask to keep intensity between 0 and 1
alpha = alpha.astype(float)/255

# Multiply the foreground with the alpha matte
foreground = cv2.multiply(alpha, foreground)

# Multiply the background with ( 1 - alpha )
background = cv2.multiply(1.0 - alpha, background)

# Add the masked foreground and background.
outImage = cv2.add(foreground, background)

# Display image
cv2.imshow("outImg", outImage/255)
cv2.waitKey(0)

void alphaBlend(Mat& foreground, Mat& background, Mat& alpha, Mat& outImage)
{
  // Find number of pixels.
    int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();
      
      // Get floating point pointers to the data matrices
        float* fptr = reinterpret_cast<float*>(foreground.data);
          float* bptr = reinterpret_cast<float*>(background.data);
            float* aptr = reinterpret_cast<float*>(alpha.data);
              float* outImagePtr = reinterpret_cast<float*>(outImage.data);
                
                // Loop over all pixesl ONCE
                  for(
                      int i = 0;
                      i < numberOfPixels;
                      i++, outImagePtr++, fptr++, aptr++, bptr++
                      )
                    {
                      *outImagePtr = (*fptr)*(*aptr) + (*bptr)*(1 - *aptr);
                      }
}
