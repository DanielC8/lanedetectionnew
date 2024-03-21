import cv2
import numpy as np

video = cv2.VideoCapture("lanevid.mp4")
leftl = []
rightl = []
count = 0
# car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    print(height, width)
    pts1 = np.float32(
        [[int(width / 2) - 40, height / 2 + 40], [int(width / 2) + 40, height / 2 + 40], [0, height], [width, height]])
    cv2.line(frame, (int(width / 2) - 40, int(height / 2) + 20), (int(width / 2) + 40, int(height / 2) + 20),
             (255, 0, 255), 10)

    pts2 = np.float32([[int((int(width / 2) - 40) / 2), 0], [(int((width / 2) + 40 + width) / 2), 0],
                       [int((int(width / 2) - 40) / 2), height], [(int((width / 2) + 40 + width) / 2), height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (width, height))

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    canny2 = cv2.Canny(gray,30,200)
    img_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    lowery = np.array([10, 90, 90], dtype="uint8")
    uppery = np.array([40, 255, 255], dtype="uint8")
    maskt = cv2.inRange(img_hsv, lowery, uppery)
    maskw = cv2.bitwise_and(gray, maskt)
    canny = cv2.Canny(maskw, 10, 20)
    lines = cv2.HoughLinesP(canny2, 1, np.pi / 180, 10, maxLineGap=50, minLineLength=5)

    slopedict = {}
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 == x2:
                slope = 99999
            else:
                slope = (y1 - y2) / (x1 - x2)
            points = ((x1,y1),(x2,y2))
            if slope >100 or slope < -100:
                slopedict[points] = slope
        posline = []
        negline = []
        for pointset in slopedict:
            xval = pointset[0][0]
            if xval > width/2:
                if posline != []:
                    if xval-width/2 < posline[0]:
                        posline = [xval-width/2,pointset,slopedict[pointset]]
                else:
                    posline = [xval - width / 2, pointset, slopedict[pointset]]
            else:
                if negline != []:
                    if xval-width/2 > negline[0]:
                        negline = [xval-width/2,pointset,slopedict[pointset]]
                else:
                    negline = [xval - width / 2, pointset, slopedict[pointset]]
        if posline != []:
            if rightl != []:
                if posline[1][0][0]-width < rightl[2]:
                    count = 0
                    x1 = posline[1][0][0]
                    y1 = posline[1][0][1]
                    x2 = posline[1][1][0]
                    y2 = posline[1][1][1]
                    slope = posline[2]
                    yintercept = y1-slope*x1
                    newx1 = -yintercept/slope
                    newx2 = -yintercept/slope

                    cv2.line(result, (int(newx1), 0), (int(newx2), height), (0, 255, 255), 10)
                    rightl = [(newx1,0),(newx2,height),x1-width]
                elif count <10:
                    count += 1
                    newx1 = rightl[0][0]
                    newx2 = rightl[1][0]
                    cv2.line(result, (int(newx1), 0), (int(newx2), height), (0, 255, 255), 10)
                else:
                    count = 0
                    x1 = posline[1][0][0]
                    y1 = posline[1][0][1]
                    x2 = posline[1][1][0]
                    y2 = posline[1][1][1]
                    slope = posline[2]
                    yintercept = y1 - slope * x1
                    newx1 = -yintercept / slope
                    newx2 = -yintercept / slope

                    cv2.line(result, (int(newx1), 0), (int(newx2), height), (0, 255, 255), 10)
                    rightl = [(newx1, 0), (newx2, height), x1 - width]
            else:
                count = 0
                x1 = posline[1][0][0]
                y1 = posline[1][0][1]
                x2 = posline[1][1][0]
                y2 = posline[1][1][1]
                slope = posline[2]
                yintercept = y1 - slope * x1
                newx1 = -yintercept / slope
                newx2 = -yintercept / slope

                cv2.line(result, (int(newx1), 0), (int(newx2), height), (0, 255, 255), 10)
                rightl = [(newx1, 0), (newx2, height), x1 - width]
        elif rightl != [] and count <10:
            count += 1
            newx1 = rightl[0][0]
            newx2 = rightl[1][0]
            cv2.line(result, (int(newx1), 0), (int(newx2), height), (0, 255, 255), 10)
        print(leftl)
        if negline != []:
            if leftl != []:
                if int(negline[1][0][0]-width) > int(leftl[2]):
                    count = 0
                    x1 = negline[1][0][0]
                    y1 = negline[1][0][1]
                    x2 = negline[1][1][0]
                    y2 = negline[1][1][1]
                    slope = negline[2]
                    yintercept = y1 - slope * x1
                    newx1 = -yintercept / slope
                    newx2 = -yintercept / slope
                    cv2.line(result, (int(newx1), 0), (int(newx2), height), (0, 255, 255), 10)
                    leftl = [(newx1, 0), (newx2, height),x1-width]
                elif count <10:
                    count +=1
                    newx1 = leftl[0][0]
                    newx2 = leftl[1][0]
                    cv2.line(result, (int(newx1), 0), (int(newx2), height), (0, 255, 255), 10)
                else:
                    count = 0
                    x1 = negline[1][0][0]
                    y1 = negline[1][0][1]
                    x2 = negline[1][1][0]
                    y2 = negline[1][1][1]
                    slope = negline[2]
                    yintercept = y1 - slope * x1
                    newx1 = -yintercept / slope
                    newx2 = -yintercept / slope
                    cv2.line(result, (int(newx1), 0), (int(newx2), height), (0, 255, 255), 10)
                    leftl = [(newx1, 0), (newx2, height), x1 - width]
            else:
                count = 0
                x1 = negline[1][0][0]
                y1 = negline[1][0][1]
                x2 = negline[1][1][0]
                y2 = negline[1][1][1]
                slope = negline[2]
                yintercept = y1 - slope * x1
                newx1 = -yintercept / slope
                newx2 = -yintercept / slope
                cv2.line(result, (int(newx1), 0), (int(newx2), height), (0, 255, 255), 10)
                leftl = [(newx1, 0), (newx2, height), x1 - width]
        elif leftl != [] and count <10:
            count += 1
            newx1 = leftl[0][0]
            newx2 = leftl[1][0]
            cv2.line(result, (int(newx1), 0), (int(newx2), height), (0, 255, 255), 10)

    if rightl != [] and leftl != []:
        cv2.line(result, (int((rightl[0][0] + leftl[0][0])/2), 0), (int((rightl[1][0] + leftl[1][0])/2), height), (0, 255, 255), 10)



    matrix = cv2.getPerspectiveTransform(pts2, pts1)
    result = cv2.warpPerspective(result, matrix, (width, height))
    cv2.imshow("output", result)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
