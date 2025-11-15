import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import math

# =========================
# Parameters (tune as needed)
# =========================
SMOOTH_N        = 7          # moving average length for S_yaw, S_pitch
YAW_THRESH      = 0.12       # threshold for LEFT/RIGHT decision
PITCH_THRESH    = 0.10       # threshold for UP/DOWN decision
HYSTERESIS      = 0.02
DRAW_TESSELLATION = True
POINT_SIZE      = 2
POINT_THICK     = -1

# Colors: BGR
COL_UP     = (255,  0,  0)   # blue
COL_DOWN   = (  0,255,  0)   # green
COL_LEFT   = (  0,  0,255)   # red
COL_RIGHT  = (  0,255,255)   # yellow
COL_CENTER = (255,255,255)   # white
COL_GRID   = (160,160,160)
COL_TEXT   = (255,255,0)

def put_text(img, text, org, scale=0.8, color=(255,255,255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

class Smoother:
    def __init__(self, n=5):
        self.qx = deque(maxlen=n)
        self.qy = deque(maxlen=n)
    def push(self, sx, sy):
        self.qx.append(sx)
        self.qy.append(sy)
    def get(self):
        if not self.qx:
            return 0.0, 0.0
        return float(np.mean(self.qx)), float(np.mean(self.qy))

def rotate_points(points, center, angle_rad):
    """
    Rotate Nx2 array of points around 'center' by angle_rad (radians).
    Positive angle = CCW in mathematical coords.
    OpenCV coords: y down, so visually it will look mirrored but
    roll compensation effect is consistent.
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    out = np.empty_like(points)
    cx, cy = center
    for i, (x, y) in enumerate(points):
        dx = x - cx
        dy = y - cy
        xr = cx + c * dx - s * dy
        yr = cy + s * dx + c * dy
        out[i] = (xr, yr)
    return out

def main():
    mp_face_mesh = mp.solutions.face_mesh
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    smoother = Smoother(SMOOTH_N)
    last_label = "CENTER"

    # Offsets captured on 'C'
    yaw_off = 0.0
    pitch_off = 0.0

    # Precompute indices for face oval from the connection list
    face_oval_connections = mp_face_mesh.FACEMESH_FACE_OVAL
    oval_indices = sorted(list({i for conn in face_oval_connections for i in conn}))

    # Landmark indices (MediaPipe FaceMesh)
    NOSE_IDX = 1
    LEFT_EYE_IDX = 33
    RIGHT_EYE_IDX = 263

    tess_conn = mp_face_mesh.FACEMESH_TESSELATION

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            label = "CENTER"
            S_yaw = 0.0
            S_pitch = 0.0

            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark

                # All points in image coordinates
                pts_xy = np.array([(lm.x * w, lm.y * h) for lm in lms], dtype=np.float32)

                # Face oval subset
                oval_pts = pts_xy[oval_indices, :]

                # Basic face ROI and center (before rotation)
                x_min = int(max(0, np.min(oval_pts[:, 0])))
                x_max = int(min(w - 1, np.max(oval_pts[:, 0])))
                y_min = int(max(0, np.min(oval_pts[:, 1])))
                y_max = int(min(h - 1, np.max(oval_pts[:, 1])))

                cx = 0.5 * (x_min + x_max)
                cy = 0.5 * (y_min + y_max)
                face_center = (cx, cy)

                # Get eye points and nose point (for roll and distances)
                P_n = pts_xy[NOSE_IDX]
                P_le = pts_xy[LEFT_EYE_IDX]
                P_re = pts_xy[RIGHT_EYE_IDX]

                # Compute roll angle using the eye line
                dx_eye = P_re[0] - P_le[0]
                dy_eye = P_re[1] - P_le[1]
                roll = math.atan2(dy_eye, dx_eye)

                # Rotate oval points and key points by -roll around face center
                all_key_points = np.vstack([oval_pts,
                                            P_n.reshape(1, 2),
                                            P_le.reshape(1, 2),
                                            P_re.reshape(1, 2)])
                all_rot = rotate_points(all_key_points, face_center, -roll)

                # Split back
                oval_rot = all_rot[:len(oval_pts)]
                P_n_rot = all_rot[len(oval_pts)]
                P_le_rot = all_rot[len(oval_pts) + 1]
                P_re_rot = all_rot[len(oval_pts) + 2]

                # From rotated oval, find boundary points
                x_min_r = np.min(oval_rot[:, 0])
                x_max_r = np.max(oval_rot[:, 0])
                y_min_r = np.min(oval_rot[:, 1])
                y_max_r = np.max(oval_rot[:, 1])

                # Boundary "four dots" (aligned with rotated nose)
                P_L = np.array([x_min_r, P_n_rot[1]])   # same y as nose
                P_R = np.array([x_max_r, P_n_rot[1]])
                P_U = np.array([P_n_rot[0], y_min_r])   # same x as nose
                P_D = np.array([P_n_rot[0], y_max_r])

                # Face width and height in rotated space
                W = np.linalg.norm(P_R - P_L)
                H = np.linalg.norm(P_D - P_U)
                W = max(W, 1e-6)
                H = max(H, 1e-6)

                # Distances nose -> boundaries (rotated, normalized)
                d_L = np.linalg.norm(P_n_rot - P_L) / W
                d_R = np.linalg.norm(P_n_rot - P_R) / W
                d_U = np.linalg.norm(P_n_rot - P_U) / H
                d_D = np.linalg.norm(P_n_rot - P_D) / H

                # Continuous scores in [-1, 1]
                denom_lr = max(d_L + d_R, 1e-6)
                denom_ud = max(d_U + d_D, 1e-6)

                S_yaw_raw   = (d_R - d_L) / denom_lr          # right positive, left negative
                S_pitch_raw = (d_D - d_U) / denom_ud          # *** FIXED: up positive, down negative ***

                # Apply calibration offsets
                S_yaw   = S_yaw_raw   - yaw_off
                S_pitch = S_pitch_raw - pitch_off

                # Smooth
                smoother.push(S_yaw, S_pitch)
                S_yaw_s, S_pitch_s = smoother.get()

                # Optional: discrete label with hysteresis for display
                yaw_thr_pos = YAW_THRESH + (HYSTERESIS if last_label == "RIGHT" else 0.0)
                yaw_thr_neg = -(YAW_THRESH + (HYSTERESIS if last_label == "LEFT"  else 0.0))
                pit_thr_pos = PITCH_THRESH + (HYSTERESIS if last_label == "UP"    else 0.0)
                pit_thr_neg = -(PITCH_THRESH + (HYSTERESIS if last_label == "DOWN" else 0.0))

                if S_yaw_s >= yaw_thr_pos:
                    label = "RIGHT"
                elif S_yaw_s <= yaw_thr_neg:
                    label = "LEFT"
                elif S_pitch_s >= pit_thr_pos:
                    label = "UP"
                elif S_pitch_s <= pit_thr_neg:
                    label = "DOWN"
                else:
                    label = "CENTER"

                last_label = label

                # ------------------------------
                # Drawing section
                # ------------------------------
                if DRAW_TESSELLATION:
                    mp_draw.draw_landmarks(
                        frame, res.multi_face_landmarks[0], tess_conn,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                    )

                # Draw oval points in original space
                for (x, y) in oval_pts.astype(int):
                    cv2.circle(frame, (int(x), int(y)), 1, (200, 200, 200), -1, lineType=cv2.LINE_AA)

                # Draw nose in original space
                cv2.circle(frame, (int(P_n[0]), int(P_n[1])), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)

                # Compute original-space versions of boundary markers for display
                bd_rot = np.vstack([P_L, P_R, P_U, P_D])
                bd_orig = rotate_points(bd_rot, face_center, roll)
                P_L_o, P_R_o, P_U_o, P_D_o = bd_orig

                for pt, col in [(P_L_o, COL_LEFT),
                                (P_R_o, COL_RIGHT),
                                (P_U_o, COL_UP),
                                (P_D_o, COL_DOWN)]:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, col, -1, lineType=cv2.LINE_AA)

                # Draw bounding box of oval
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), COL_GRID, 1)

                # Debug text: raw and smoothed continuous scores
                put_text(frame, f"S_yaw_raw:   {S_yaw_raw:+.3f}",   (10, 28), 0.6, COL_TEXT, 1)
                put_text(frame, f"S_pitch_raw: {S_pitch_raw:+.3f}", (10, 48), 0.6, COL_TEXT, 1)
                put_text(frame, f"S_yaw_s:     {S_yaw_s:+.3f}",     (10, 68), 0.6, (220,220,220), 1)
                put_text(frame, f"S_pitch_s:   {S_pitch_s:+.3f}",   (10, 88), 0.6, (220,220,220), 1)

            # Big label (discrete)
            put_text(frame, label, (10, h - 20), 1.2, (0,255,255), 3)
            put_text(frame, "C: calibrate center   Q: quit", (w-360, 24), 0.55, (200,200,200), 1)

            cv2.imshow("Head Direction (Nose + Boundaries, Roll Compensated)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('c'):
                # Store current smoothed scores as neutral offsets
                yaw_off, pitch_off = smoother.get()
                print(f"[Calibrated] yaw_offset={yaw_off:+.3f}, pitch_offset={pitch_off:+.3f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
