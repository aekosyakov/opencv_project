import cv2
import numpy as np

def blend_transparent(face_img, overlay_t_img):
  # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane
    
    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask
    
    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)
    
    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
    
    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))

def addMask(imageStr):
      cv_image = cv2.imread(imageStr)
      background_width = np.size(cv_image, 1)
      #              new_h = np.size(face, 0)
      # Load the face detection cascade file.
      face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
      nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
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
      background_ratio = float()

      # Declare background size and padding.
      background_size = 1100

      padding_top = int(((background_size - foreground_size) / 3) * 2)
      padding_bottom = int(background_size - padding_top)
      padding_left = int((background_size - foreground_size) / 2)
      padding_right = int((background_size - foreground_size) / 2)
      # Capture selfie image in OpenCV.
     

      # Load our overlay image: mustache.png
      imgMustache = cv2.imread('mustache.png',-1)

      # Create the mask for the mustache
      orig_mask = imgMustache[:,:,3]

      # Create the inverted mask for the mustache
      orig_mask_inv = cv2.bitwise_not(orig_mask)

      # Convert mustache image to BGR
      # and save the original image size (used later when re-sizing the image)
      imgMustache = imgMustache[:,:,0:3]
      origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]

      # Create greyscale image from the video feed
      gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

      # Find that face.
      faces = face_cascade.detectMultiScale(
                                            cv_image,
                                            scaleFactor=1.1,
                                            minNeighbors=3,
                                            minSize=(30, 30),
                                            flags=0
                                            )



      for (nx,ny,nw,nh) in faces:
        # Un-comment the next line for debug (draw box around all faces)
        # face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
              roi_gray  = gray[ny:ny+nh, nx:nx+nw]
              roi_color = cv_image[ny:ny+nh, nx:nx+nw]
            
              # Un-comment the next line for debug (draw box around the nose)
              #cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
            
              # The mustache should be three times the width of the nose
              mustacheWidth =  nw #int(3 * nw)
              mustacheHeight = nh #int(mustacheWidth * origMustacheHeight / origMustacheWidth)
              print(mustacheWidth)
              print(nw)
              print(mustacheHeight)
              print(nh)
              # Center the mustache on the bottom of the nose
              x1 = nx - (mustacheWidth/4)
              x2 = nx + nw + (mustacheWidth/4)
              y1 = ny + nh - (mustacheHeight/2)
              y2 = ny + nh + (mustacheHeight/2)
              
              # Check for clipping
              if x1 < 0:
                x1 = 0
              if y1 < 0:
                y1 = 0
              if x2 > nw:
                x2 = nw
              if y2 > nh:
                y2 = nh
              
              # Re-calculate the width and height of the mustache image
              mustacheWidth  = x2 - x1
              mustacheHeight = y2 - y1
              print(mustacheWidth)
              print(mustacheHeight)

              # Re-size the original image and the masks to the mustache sizes
              # calcualted above
              mustache = cv2.resize(imgMustache, (int(mustacheWidth),int(mustacheHeight)), interpolation = cv2.INTER_AREA)
              mask = cv2.resize(orig_mask, (int(mustacheWidth),int(mustacheHeight)), interpolation = cv2.INTER_AREA)
              mask_inv = cv2.resize(orig_mask_inv, (int(mustacheWidth),int(mustacheHeight)), interpolation = cv2.INTER_AREA)
              
              # take ROI for mustache from background equal to size of mustache image
              roi = roi_color[int(y1):int(y2), int(x1):int(x2)]
              
              # roi_bg contains the original image only where the mustache is not
              # in the region that is the size of the mustache.
              roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
              
              # roi_fg contains the image of the mustache only where the mustache is
              roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
              
              # join the roi_bg and roi_fg
              dst = cv2.add(roi_bg,roi_fg)
              
              # place the joined image, saved to dst back over the original image
              roi_color[int(y1):int(y2), int(x1):int(x2)] = dst
              cv2.imwrite('cropface/face.jpg', cv_image)
              # Detect a nose within the region bounded by each face (the ROI)
              nose = nose_cascade.detectMultiScale(roi_gray)
              print('face ' + str(x))

#              for (nx,ny,nw,nh) in nose:
#                  # Un-comment the next line for debug (draw box around the nose)
#                  #cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
#                
#                  # The mustache should be three times the width of the nose
#                  mustacheWidth =  int(3 * nw)
#                  mustacheHeight = int(mustacheWidth * origMustacheHeight / origMustacheWidth)
#                  
#                  # Center the mustache on the bottom of the nose
#                  x1 = nx - (mustacheWidth/4)
#                  x2 = nx + nw + (mustacheWidth/4)
#                  y1 = ny + nh - (mustacheHeight/2)
#                  y2 = ny + nh + (mustacheHeight/2)
#                  
#                  # Check for clipping
#                  if x1 < 0:
#                    x1 = 0
#                  if y1 < 0:
#                    y1 = 0
#                  if x2 > w:
#                    x2 = w
#                  if y2 > h:
#                    y2 = h
#                  
#                  # Re-calculate the width and height of the mustache image
#                  mustacheWidth = x2 - x1
#                  mustacheHeight = y2 - y1
#                  
#                  # Re-size the original image and the masks to the mustache sizes
#                  # calcualted above
#                  mustache = cv2.resize(imgMustache, (int(mustacheWidth),int(mustacheHeight)), interpolation = cv2.INTER_AREA)
#                  mask = cv2.resize(orig_mask, (int(mustacheWidth),int(mustacheHeight)), interpolation = cv2.INTER_AREA)
#                  mask_inv = cv2.resize(orig_mask_inv, (int(mustacheWidth),int(mustacheHeight)), interpolation = cv2.INTER_AREA)
#                  
#                  # take ROI for mustache from background equal to size of mustache image
#                  roi = roi_color[int(y1):int(y2), int(x1):int(x2)]
#                  
#                  # roi_bg contains the original image only where the mustache is not
#                  # in the region that is the size of the mustache.
#                  roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
#
#                  # roi_fg contains the image of the mustache only where the mustache is
#                  roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
#
#                  # join the roi_bg and roi_fg
#                  dst = cv2.add(roi_bg,roi_fg)
#
#                  # place the joined image, saved to dst back over the original image
#                  roi_color[int(y1):int(y2), int(x1):int(x2)] = dst
#                  cv2.imwrite('cropface/face.jpg', cv_image)
#                  print('save')
              break



addMask('people1.jpg')
