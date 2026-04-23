import cv2
import numpy as np

def smooth(prev, current, alpha=0.7):
    if prev is None:
        return current
    return alpha * prev + (1 - alpha) * current


def flow_to_direction(flow):
    if flow is None:
        return "N/A"

    dx, dy = flow
    if abs(dx) < 0.3 and abs(dy) < 0.3:
        return "Static"

    if abs(dx) > abs(dy):
        return "E" if dx > 0 else "W"
    else:
        return "S" if dy > 0 else "N"


def draw_zones(frame, zones):
    """
    Draw zone polygons and names on the frame
    """
    for z in zones:
        pts = np.array(z["points"], np.int32).reshape((-1, 1, 2))

        # Draw zone boundary
        cv2.polylines(
            frame,
            [pts],
            isClosed=True,
            color=(255, 0, 0),
            thickness=2
        )

        # Draw zone label
        x, y = pts[0][0]
        cv2.putText(
            frame,
            z["name"],
            (int(x) + 10, int(y) + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )
