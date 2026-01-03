import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
import threading

EYE_CLOSED_FRAMES = 20
EAR_THRESHOLD = 0.25

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)

closed_frames = 0
alarm_on = False


def sound_alarm():
    playsound("alarm.wav")


def eye_aspect_ratio(eye):
    vertical1 = np.linalg.norm(eye[1] - eye[5])
    vertical2 = np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            left_eye = np.array([
                [int(face_landmarks.landmark[i].x * w),
                 int(face_landmarks.landmark[i].y * h)]
                for i in LEFT_EYE
            ])

            right_eye = np.array([
                [int(face_landmarks.landmark[i].x * w),
                 int(face_landmarks.landmark[i].y * h)]
                for i in RIGHT_EYE
            ])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2

            if ear < EAR_THRESHOLD:
                closed_frames += 1

                if closed_frames >= EYE_CLOSED_FRAMES:
                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT!",
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3
                    )

                    if not alarm_on:
                        alarm_on = True
                        threading.Thread(
                            target=sound_alarm,
                            daemon=True
                        ).start()
            else:
                closed_frames = 0
                alarm_on = False

            for point in np.concatenate((left_eye, right_eye)):
                cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
