

import cv2
import numpy as np
import math



REFERENCE_ANGLE = 90.0      
SMOOTHING_ALPHA = 0.25      


SKIN_LOWER = np.array([0,  20,  70], dtype=np.uint8)
SKIN_UPPER = np.array([20, 255, 255], dtype=np.uint8)


MIN_CONTOUR_AREA = 3000




def detect_skin_mask(frame):
   
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SKIN_LOWER, SKIN_UPPER)

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def find_arm_contour(mask):
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
        return None
    return largest


def fit_skeleton_points(contour, n_points=5):
  
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx  = cv2.approxPolyDP(contour, epsilon, True)
    pts     = approx[:, 0, :]

   
    pts = pts[pts[:, 1].argsort()]

    if len(pts) < 3:
        return None

    
    y_vals  = np.linspace(pts[0, 1], pts[-1, 1], n_points, dtype=int)
    centers = []
    for y in y_vals:
       
        nearby = pts[np.abs(pts[:, 1] - y) < (pts[-1, 1] - pts[0, 1]) / n_points + 5]
        if len(nearby):
            cx = int(np.mean(nearby[:, 0]))
            centers.append((cx, int(y)))

    return centers if len(centers) >= 3 else None


def find_elbow_point(centers):
   
    if len(centers) < 3:
        return None, None, None

    max_dev  = -1
    elbow_i  = len(centers) // 2  

    for i in range(1, len(centers) - 1):
        p0 = np.array(centers[i - 1], dtype=float)
        p1 = np.array(centers[i],     dtype=float)
        p2 = np.array(centers[i + 1], dtype=float)

        
        line_vec  = p2 - p0
        line_len  = np.linalg.norm(line_vec)
        if line_len < 1e-6:
            continue
        dev = abs(np.cross(line_vec, p0 - p1)) / line_len

        if dev > max_dev:
            max_dev  = dev
            elbow_i  = i

    return centers[elbow_i], centers[0], centers[-1]


def calc_angle(a, vertex, b):
   
    va = np.array(a, dtype=float) - np.array(vertex, dtype=float)
    vb = np.array(b, dtype=float) - np.array(vertex, dtype=float)

    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na < 1e-6 or nb < 1e-6:
        return None

    cos_theta = np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def angle_to_score(angle):
   
    return round(angle - REFERENCE_ANGLE, 1)


def draw_overlay(frame, centers, elbow, score, angle, mask):
   
    h, w = frame.shape[:2]

    
    overlay = frame.copy()
    colored_mask = np.zeros_like(frame)
    colored_mask[mask > 0] = (0, 200, 80)
    cv2.addWeighted(colored_mask, 0.25, overlay, 0.75, 0, overlay)

    
    if centers and len(centers) >= 2:
        for i in range(len(centers) - 1):
            cv2.line(overlay, centers[i], centers[i + 1], (255, 200, 0), 2)
        for pt in centers:
            cv2.circle(overlay, pt, 5, (255, 200, 0), -1)

  
    if elbow:
        cv2.circle(overlay, elbow, 12, (0, 80, 255), -1)
        cv2.circle(overlay, elbow, 14, (255, 255, 255), 2)

   
    panel_h = 110
    cv2.rectangle(overlay, (0, 0), (280, panel_h), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, 0), (280, panel_h), (60, 60, 60),  1)

    if angle is not None:
        color_score = (0, 220, 80) if score >= 0 else (80, 180, 255)
        sign        = "+" if score > 0 else ""

        cv2.putText(overlay, f"Angle: {angle:.1f} deg",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2)
        cv2.putText(overlay, f"Score: {sign}{score}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color_score, 3)
        cv2.putText(overlay, "(0 = 90 deg reference)",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)
    else:
        cv2.putText(overlay, "No arm detected",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 120, 255), 2)


    cv2.putText(overlay, "Q = quit  |  R = recalibrate skin",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)

    return overlay



def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("خطا: وب‌کم پیدا نشد.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    smoothed_angle = None
    print("شروع شد — Q برای خروج")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   

        mask     = detect_skin_mask(frame)
        contour  = find_arm_contour(mask)
        centers  = fit_skeleton_points(contour) if contour is not None else None
        elbow, p_top, p_bot = find_elbow_point(centers) if centers else (None, None, None)

        angle = score = None
        if elbow and p_top and p_bot:
            angle = calc_angle(p_top, elbow, p_bot)
            if angle is not None:
                
                if smoothed_angle is None:
                    smoothed_angle = angle
                else:
                    smoothed_angle = (SMOOTHING_ALPHA * angle +
                                      (1 - SMOOTHING_ALPHA) * smoothed_angle)
                angle = round(smoothed_angle, 1)
                score = angle_to_score(angle)

        result = draw_overlay(frame, centers, elbow, score, angle, mask)
        cv2.imshow("Elbow Tracker — OpenCV", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
           
            smoothed_angle = None
            print("ریست شد.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()