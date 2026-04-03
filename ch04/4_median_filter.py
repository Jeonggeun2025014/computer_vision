import cv2

img = cv2.imread('ch04_salt_pepper_noise.png', cv2.IMREAD_GRAYSCALE)

median = cv2.medianBlur(img, 5)

cv2.imshow('Original', img)
cv2.imshow('Median Filter', median)
cv2.waitKey(0)
cv2.destroyAllWindows()