from lib.detector import VehicleDetector

import cv2

if __name__ == '__main__':

    det = VehicleDetector()

    im = cv2.imread('./car.jpg')
    result = det.detect(im)

    cv2.imshow('a', result)
    cv2.waitKey(0)

    cv2.imwrite('./result.png', result)

    cv2.destroyAllWindows()
