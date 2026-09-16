import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay

# ---------------- CONFIG ----------------
IMAGE_PATH = "avatar2.png"
CAMERA_INDEX = 0
SMOOTHING = 0.1
# ----------------------------------------

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Load avatar image
avatar = cv2.imread(IMAGE_PATH)
avatar = cv2.resize(avatar, (512, 512))
h, w = avatar.shape[:2]

print(f"Avatar image size: {w}x{h}")

# Detect landmarks on avatar (once)
with mp_face.FaceMesh(static_image_mode=True) as static_mesh:

    rgb = cv2.cvtColor(avatar, cv2.COLOR_BGR2RGB)
    result = static_mesh.process(rgb)

    if not result.multi_face_landmarks:
        raise RuntimeError("No face found in avatar image")

    avatar_landmarks = np.array([
        [int(lm.x * w), int(lm.y * h)]
        for lm in result.multi_face_landmarks[0].landmark
    ])

# Delaunay triangulation
print(avatar_landmarks)
print(avatar_landmarks.shape)
tri = Delaunay(avatar_landmarks)

print("Avatar landmarks and triangulation ready.")
print(tri.simplices)
print(tri.simplices.shape)

def warp_triangle(img1, img2, t1, t2):
    t1 = np.float32(t1)
    t2 = np.float32(t2)

    r1 = cv2.boundingRect(t1)
    r2 = cv2.boundingRect(t2)
    #cv2.rectangle(img2, (r2[0], r2[1]), (r2[0]+r2[2], r2[1]+r2[3]), (0,255,0), 1)
    #cv2.rectangle(img1, (r1[0], r1[1]), (r1[0]+r1[2], r1[1]+r1[3]), (255,0,0), 1)

    # Clamp to image bounds
    h, w = img2.shape[:2]
    #print(h, w)
    if (r2[0] < 0 or r2[1] < 0 or
        r2[0] + r2[2] > w or
        r2[1] + r2[3] > h):
        return

    t1_rect = []
    t2_rect = []

    for i in range(3):
        t1_rect.append((t1[i][0] - r1[0], t1[i][1] - r1[1]))
        t2_rect.append((t2[i][0] - r2[0], t2[i][1] - r2[1]))

    t1_rect = np.float32(t1_rect)
    t2_rect = np.float32(t2_rect)

    img1_rect = img1[r1[1]:r1[1]+r1[3],
                     r1[0]:r1[0]+r1[2]]

    if img1_rect.size == 0:
        return

    size = (r2[2], r2[3])
    warp_mat = cv2.getAffineTransform(t1_rect, t2_rect)
    img2_rect = cv2.warpAffine(
        img1_rect,
        warp_mat,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0))

    roi = img2[r2[1]:r2[1]+r2[3],
               r2[0]:r2[0]+r2[2]]

    if roi.shape != img2_rect.shape:
        return

    img2[r2[1]:r2[1]+r2[3],
         r2[0]:r2[0]+r2[2]] = (
        roi * (1-mask) + img2_rect *(mask)
    )

# Webcam
cap = cv2.VideoCapture(CAMERA_INDEX)

prev_landmarks = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    h, w = frame.shape[:2]

    if result.multi_face_landmarks:
        landmarks = np.array([
            [int(lm.x * w), int(lm.y * h)]
            for lm in result.multi_face_landmarks[0].landmark
        ])

        if prev_landmarks is None:
            prev_landmarks = landmarks

        #print(landmarks)
        #print(landmarks.shape)
        #print("hi")
        landmarks = (SMOOTHING * prev_landmarks + (1 - SMOOTHING) * landmarks).astype(int)

        prev_landmarks = landmarks
        #print(landmarks)
        #print(landmarks.shape)
        #print("bye")
    

        #output = np.zeros_like(avatar) # black background
        output = frame.copy()

        for tri_indices in tri.simplices:
            t1 = avatar_landmarks[tri_indices]
            t2 = landmarks[tri_indices]
            warp_triangle(avatar, output, t1, t2)
            #cv2.line(output, landmarks[tri_indices[0]], landmarks[tri_indices[1]], (0, 255, 0), 1)
            #cv2.line(output, landmarks[tri_indices[1]], landmarks[tri_indices[2]], (0, 255, 0), 1)
            #cv2.line(output, landmarks[tri_indices[2]], landmarks[tri_indices[0]], (0, 255, 0), 1)

        cv2.imshow("Live Output", output)
        cv2.imshow("Webcam", frame)

    else:
        cv2.imshow("Live Avatar", avatar)
        cv2.imshow("Webcam", frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
