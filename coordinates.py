import cv2

VIDEO = "crowd_analytics/assets/final.mp4"

points = []

cap = cv2.VideoCapture(VIDEO)
ret, frame = cap.read()
cap.release()

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: [{x}, {y}]")
        points.append((x, y))

WINDOW = "Click to get coordinates (press q to quit)"

cv2.namedWindow(WINDOW, cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(WINDOW, 540, 960)
cv2.imshow(WINDOW, frame)
cv2.setMouseCallback(WINDOW, mouse_callback)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
