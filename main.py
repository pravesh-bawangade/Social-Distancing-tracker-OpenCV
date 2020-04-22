from imutils.object_detection import non_max_suppression
import numpy as np
import cv2


def main():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    cap = cv2.VideoCapture('video/output.avi')
    frame_width = int(1280)
    frame_height = int(720)

    out = cv2.VideoWriter('video/output_det.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

    while True:

        ret, image = cap.read()
        if ret:
            image = cv2.resize(image,(1280,720))
            (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                                    padding=(4, 4), scale=1.05)

            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.35)
            cJoin = np.zeros((pick.shape[0],2),dtype=int)
            count=0
            for (xA, yA, xB, yB) in pick:
                x_centre, y_centre = int((xB+xA) / 2), int((yB+yA)/2)
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
                cv2.circle(image, center=(x_centre,y_centre),radius=5,color=(255,0,0),thickness=-1)
                cJoin[count] = [x_centre,y_centre]
                count+=1

            for (cX,cY) in cJoin:
                for (cX_2, cY_2) in cJoin:
                    if (cX,cY) == (cX_2,cY_2):
                        pass
                    else:
                        lineLenth = ((((cX_2-cX)**2) + ((cY_2-cY)**2))**(1/2))
                        if lineLenth < 90:
                            cv2.line(image,(cX,cY),(cX_2,cY_2),(0,0,255),thickness=2)
                        elif 120 > lineLenth > 90:
                            cv2.line(image,(cX,cY),(cX_2,cY_2),(0,255,255),thickness=2)

            cv2.imshow("Image", image)
            out.write(image)
            key = cv2.waitKey(5) & 0xFF
            if key == ord("q"):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
