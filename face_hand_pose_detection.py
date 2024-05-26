import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5) as face_mesh, \
    mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
    mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(image_rgb)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, 
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )

    hand_results = hands.process(image_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    pose_results = pose.process(image_rgb)
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('MediaPipe Combined Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()