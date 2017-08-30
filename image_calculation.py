import cv2
from matplotlib import pyplot as plt

img = cv2.imread("portrait.jpg")
img_back = cv2.imread("pushkin.jpg")

fgbg = cv2.createBackgroundSubtractorMOG2()
fgmask = fgbg.apply(img_back)
cv2.imwrite('crop_portrait.jpg',fgmask)

res_img = cv2.resize(img,(500, 500), interpolation = cv2.INTER_CUBIC)
res_img_back = cv2.resize(img_back,(500, 500), interpolation = cv2.INTER_CUBIC)

#result_image = cv2.add(res_img, res_img_back)
result_image = res_img + res_img_back
plt.imshow(result_image)
plt.show()

