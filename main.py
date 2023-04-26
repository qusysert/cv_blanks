import numpy as np
import cv2
import matplotlib.pyplot as plt

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def stream_webcam():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, img = cap.read()

        h, w, _ = img.shape

        width = 800
        height = int(width * (h / w))
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

        corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)

        detected_markers = aruco_display(corners, ids, rejected, img)

        cv2.imshow("Image", detected_markers)
        cv2.imshow("Trimmed", trim_writing_area(img, corners, ids))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cap.release()
            break


def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        for i, markerCorner in enumerate(corners):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            points = {
                0: bottomRight,
                1: bottomLeft,
                2: topLeft,
                3: topRight
            }

            # Draw the marker outline
            contour = np.array([points[j] for j in range(4)], dtype=np.int32)
            cv2.drawContours(image, [contour], 0, (0, 255, 0), 2)

            # Draw the marker ID
            markerID = int(ids[i])
            cv2.putText(image, str(markerID), (int(topLeft[0]), int(topLeft[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            # Draw the circle at the corner point
            cX, cY = points[markerID]
            cv2.circle(image, (int(cX), int(cY)), 4, (0, 0, 255), -1)

            print("[Inference] ArUco marker ID: {}. Coordinates are x: {}, y: {}".format(markerID, cX, cY))

    return image


def stretch_polygon_to_rectangle(points, image):
    # Define the dimensions of the output rectangle
    width = 740
    height = 900

    # Define the corners of the output rectangle
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Convert the input points to a numpy array
    src = np.array(points, dtype=np.float32)

    # Reshape the input points to the correct format
    src = src.reshape((4, 2))

    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Apply the perspective transform to the image
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped


def corner_arucos_to_list(corners, ids):
    mm = {}
    for i, markerCorner in enumerate(corners):
        marker_corners = markerCorner.reshape((4, 2))
        print("marker corner:", markerCorner)
        marker_id = int(ids[i])
        mm[marker_id] = marker_corners
    return mm


def trim_service_area(img, corners, ids):

    if 2 in ids and 3 in ids:
        height, width, _ = img.shape
        mm = corner_arucos_to_list(corners, ids)
        polygon = np.array([mm[3][3], mm[2][2], (width - 1, height - 1), (0, height - 1)])
        polygon = polygon.astype(np.float32)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, np.int32([polygon]), (255, 255, 255))

        # Apply the mask to the image
        masked_img = cv2.bitwise_and(img, mask)

        # Get the bounding rectangle of the polygon
        x, y, w, h = cv2.boundingRect(np.int32([polygon]))

        # Crop the image using the bounding rectangle
        crop_img = masked_img[y:y + h, x:x + w]
        return crop_img
    else:
        return img


def trim_writing_area(img, corners, ids):
    print("Amount of found corners: ", len(corners))
    if len(corners) == 4:
        mm = corner_arucos_to_list(corners, ids)
        polygon = np.array([mm[0][3], mm[1][2], mm[2][1], mm[3][0]])
        polygon = polygon.astype(np.float32)
        # Stretch the polygon to a rectangle
        warped = stretch_polygon_to_rectangle(polygon, img)
        return warped
    else:
        return img


print("hi")
aruco_type = "DICT_4X4_50"

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters()

img = cv2.imread("blanks/blank.png")

h, w, _ = img.shape
width = 700
height = int(width * (h / w))
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)

detected_markers = aruco_display(corners, ids, rejected, img)
cv2.imshow("Image", detected_markers)
# cv2.imshow("Trimmed", trim_service_area(img, corners, ids))
# stream_webcam()
img = trim_service_area(img, corners, ids)
gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
th1, img_bin = cv2.threshold(gray_scale, 150, 225, cv2.THRESH_BINARY)
img_bin = ~img_bin


### selecting min size as 15 pixels
line_min_width = 15
kernal_h = np.ones((1, line_min_width), np.uint8)
kernal_v = np.ones((line_min_width, 1), np.uint8)

img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
img_bin_final = img_bin_h | img_bin_v
# plt.imshow(img_bin_final, cmap='gray')
# plt.show()

_, labels, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
for x, y, w, h, area in stats[2:]:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("S", img)
cv2.waitKey(0)
