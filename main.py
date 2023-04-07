import numpy as np
import cv2

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

def trim_by_corners(img, corners, ids):
    mm = {}
    if len(corners) == 4:
        for i, markerCorner in enumerate(corners):
            marker_corners = markerCorner.reshape((4, 2))
            markerID = int(ids[i])
            mm[markerID] = marker_corners
        polygon = np.array([mm[0][3], mm[1][2], mm[2][1], mm[3][0]])
        polygon = polygon.astype(np.float32)

        # Stretch the polygon to a rectangle
        warped = stretch_polygon_to_rectangle(polygon, img)

        return warped

    else:
        return img


aruco_type = "DICT_4X4_50"

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

arucoParams = cv2.aruco.DetectorParameters()

img = cv2.imread("blanks/blank_aruco.png")

h, w, _ = img.shape

width = 700
height = int(width * (h / w))
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)

detected_markers = aruco_display(corners, ids, rejected, img)

cv2.imshow("Image", detected_markers)
cv2.imshow("Trimmed", trim_by_corners(img, corners, ids))


# cap = cv2.VideoCapture(0)
#
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#
# while cap.isOpened():
#     ret, img = cap.read()
#
#     h, w, _ = img.shape
#
#     width = 800
#     height = int(width * (h / w))
#     img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
#
#     corners, ids, rejected = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
#
#     detected_markers = aruco_display(corners, ids, rejected, img)
#
#     cv2.imshow("Image", detected_markers)
#     cv2.imshow("Trimmed", trim_by_corners(img, corners, ids))
#
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break


cv2.waitKey(0)
cv2.destroyAllWindows()
# cap.release()
