

if __name__ == '__main__':

    from lib.detector import VehicleDetector

    import cv2

    cap = cv2.VideoCapture('E:\\cut.mp4')

    video_width = int(cap.get(3))
    video_height = int(cap.get(4))
    fps = int(cap.get(5)/2)
    det = VehicleDetector()
    print(fps)

    while True:

        _, im = cap.read()

        if im is None:
            break

        raw = im.copy()
        result = det.detect(im)
        cv2.imshow('a', result)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
