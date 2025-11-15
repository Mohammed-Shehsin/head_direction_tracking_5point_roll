# Head Direction Tracking (5-Point Boundary + Roll Compensation)

This project implements a real-time head orientation tracking system using  
**MediaPipe FaceMesh**, **OpenCV**, and a custom **5-point boundary method**  
enhanced with **roll compensation**.

It provides:

- Continuous yaw and pitch scores in **[-1, 1]**
- Optional discrete states: **LEFT / RIGHT / UP / DOWN / CENTER**

---

# 1. Demo

<p align="center">
  <img src="assets/ezgif-483684fde172f006.gif" width="250"/>
</p>


---

# 2. Method Overview

This geometric method uses five key reference points:

- 1 nose point  
- 2 eye points  
- Face oval point set  
- 4 derived boundary points (Left, Right, Up, Down)  

A roll-compensation step rotates the entire face to an upright coordinate system,
ensuring stable LEFT/RIGHT and UP/DOWN detection even when the head is tilted.

---

# 3. Landmark Definitions

From MediaPipe FaceMesh:

- **Nose tip**: \( P_n \) — index **1**
- **Left eye outer corner**: \( P_{LE} \) — index **33**
- **Right eye outer corner**: \( P_{RE} \) — index **263**
- **Face oval points**: all indices in `FACEMESH_FACE_OVAL`

The following boundary points are derived from the oval:

- \( P_L \): minimum x  
- \( P_R \): maximum x  
- \( P_U \): minimum y  
- \( P_D \): maximum y  

The face center \( C \) is taken as the center of the oval bounding box.

---

# 4. Roll Compensation

Roll represents the “ear-to-shoulder” head tilt.  
We compute it using the eye line:

$$
 \theta_{\text{roll}} =
\arctan2
\bigl(
y_{RE} - y_{LE},\;
x_{RE} - x_{LE}
\bigr)
$$

All relevant points are rotated by **\(-\theta_{\text{roll}}\)** around the face center \( C \):

$$
P' = R(-\theta_{\text{roll}})\,\bigl(P - C\bigr) + C
$$

After this, the coordinate system becomes **upright**, making the vertical and horizontal
distances consistent regardless of head tilt.

---

# 5. Normalized Distances

Compute effective face width and height:

$$
W = \| P_R - P_L \|
$$

$$
H = \| P_D - P_U \|
$$

Normalized distances from the nose to each boundary:

$$
d_L = \frac{\|P_n - P_L\|}{W},
\qquad
d_R = \frac{\|P_n - P_R\|}{W}
$$

$$
d_U = \frac{\|P_n - P_U\|}{H},
\qquad
d_D = \frac{\|P_n - P_D\|}{H}
$$

Because distances are divided by width/height, this method is  
**scale-invariant** — face size or distance to camera do not affect results.

---

# 6. Continuous Yaw and Pitch Scores

## Horizontal (Yaw)

$$
S_{\text{yaw}}=
\frac{d_R - d_L}{d_R + d_L}
$$

- \( S_{\text{yaw}} > 0 \) → head turned **right**  
- \( S_{\text{yaw}} < 0 \) → head turned **left**

## Vertical (Pitch)

$$
S_{\text{pitch}}=
\frac{d_D - d_U}{d_D + d_U}
$$

- \( S_{\text{pitch}} > 0 \) → head tilted **up**  
- \( S_{\text{pitch}} < 0 \) → head tilted **down**

The smoothed versions:

- `S_yaw_s`  
- `S_pitch_s`  

are obtained by moving-average filtering + calibration offset removal.

---

# 7. Discrete Direction Label (Optional)

From the smoothed continuous scores, thresholds classify the direction:

- If `S_yaw_s >= +YAW_THRESH` → RIGHT  
- If `S_yaw_s <= -YAW_THRESH` → LEFT  
- If `S_pitch_s >= +PITCH_THRESH` → UP  
- If `S_pitch_s <= -PITCH_THRESH` → DOWN  
- Otherwise → CENTER  

Hysteresis prevents rapid switching (flickering).

This is mainly for visualization;  
the continuous scores are recommended for real control tasks.

---

# 8. Processing Pipeline (Per Frame)

1. Capture frame from webcam  
2. Run MediaPipe FaceMesh  
3. Extract:
   - Nose  
   - Eyes  
   - Face oval  
4. Compute roll angle \( \theta_{\text{roll}} \)  
5. Rotate points by \( -\theta_{\text{roll}} \)  
6. Derive boundaries \( P_L, P_R, P_U, P_D \)  
7. Compute \( W, H \)  
8. Compute normalized distances \( d_* \)  
9. Compute scores \( S_{\text{yaw}}, S_{\text{pitch}} \)  
10. Smooth and calibrate → `S_yaw_s`, `S_pitch_s`  
11. Optionally classify (LEFT/RIGHT/UP/DOWN/CENTER)  
12. Render visualization  

---

# 9. Installation

Install dependencies:

```
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
opencv-python
mediapipe
numpy
```

---

# 10. Running

```
head_direction_tracking_5point_roll.py
```

Controls:

- **C** → calibrate CENTER  
- **Q** → quit  

---

# 11. Using Continuous Output

The main usable control signals:

- `S_yaw_s`   ∈ [-1, 1]  
- `S_pitch_s` ∈ [-1, 1]  

## Example: Servo Mapping

```python
servo_yaw   = 90 + S_yaw_s   * 45
servo_pitch = 90 + S_pitch_s * 30
```

## Example: Cursor Movement Concept

```python
dx = int(S_yaw_s * 20)
dy = int(-S_pitch_s * 20)
```

---

# 12. Folder Structure

```
head_direction_tracking_5point_roll/
│
├── head_direction_tracking_5point_roll.py
├── README.md
├── README.txt
├── requirements.txt
├── .gitignore
│
└── assets/
    └── ezgif-483684fde172f006.gif
```

---

# 13. Author

**Mohammed Shehsin Thamarachalil Abdulresak**  
Robotics & Automation Engineering  
Poznań University of Technology, Poland
