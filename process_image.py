import cv2

def process_Th1(lis_img):
    img = cv2.imread(lis_img[0])
    img2 = cv2.imread(lis_img[1])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(img, img2)
    return diff

def process_Th2(lis_img):
    img = cv2.imread(lis_img[0])
    img2 = cv2.imread(lis_img[1])
    img_new = img2.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img, (5, 5), 2)
    img2_blur = cv2.GaussianBlur(img2, (5, 5), 2)

    diff = cv2.absdiff(img_blur, img2_blur)
    diff = cv2.medianBlur(diff, 5)

    canny = cv2.Canny(diff, 150, 200)
    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_new, contours, -1, (0,255,0), 2)

    return img_new
