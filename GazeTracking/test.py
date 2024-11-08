import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

LEFT_EYE_POINTS = [464, 413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453]
RIGHT_EYE_POINTS = [226, 113, 225, 224, 223, 222, 221, 189, 244, 233, 232, 231, 230, 229, 228, 31]

points = RIGHT_EYE_POINTS

# Use Face Mesh with default parameters
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect face landmarks
        results = face_mesh.process(image)

        # Convert the image color back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmarks on the image
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # region = np.array([(int(face_landmarks.landmark[point].x * image.shape[1]), 
                #                     int(face_landmarks.landmark[point].y * image.shape[0])) 
                #                    for point in points])
                
                # for i in range(region.shape[0]):
                #     cv2.circle(image, region[i, :].astype(np.int32), radius=1, color=(0, 255, 0), thickness=-1)
                
                for i in range(len(face_landmarks.landmark)):
                    
                    # print(f"x: {face_landmarks.landmark[i].x}\ny: {face_landmarks.landmark[i].y}\n\n")
                    
                    cv2.circle(image, (int(face_landmarks.landmark[i].x * image.shape[1]),
                                        int(face_landmarks.landmark[i].y * image.shape[0])),
                                radius=1, color=(0, 255, 0), thickness=-1)
                
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                # mp_drawing.draw_landmarks(
                #     image=image,
                #     landmark_list=face_landmarks,
                #     connections=mp_face_mesh.FACEMESH_IRISES,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # Display the resulting frame
        cv2.imshow('MediaPipe Face Mesh', image)

        # Break the loop on 'q' key press
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
# cv2.destroyAllWindows()