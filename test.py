import mediapipe as mp
import cv2
import mediapipe as mp
img_arms_raised = cv2.imread('arms_raised.jpg')
img_head_tilt = cv2.imread('head_tilt.jpeg')
img_wave = cv2.imread('wave.jpg')
img_default = cv2.imread('default.jpg')
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = 'pose_landmarker.task'

latest_result = [None]

def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    latest_result[0] = result

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        landmarker.detect_async(mp_image, timestamp_ms)
        if latest_result[0] and latest_result[0].pose_landmarks:
            for pose_landmarks in latest_result[0].pose_landmarks:
                for connection in mp.tasks.vision.PoseLandmarkerResult.__mro__:
                    pass

        # Draw each landmark as a circle
            for i, lm in enumerate(pose_landmarks):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if latest_result[0] and latest_result[0].pose_landmarks:
            landmarks = latest_result[0].pose_landmarks[0]
            if landmarks[15].y < landmarks[11].y and landmarks[16].y < landmarks[12].y:
                cv2.imshow('RedPanda', img_arms_raised)
            elif abs(landmarks[8].y - landmarks[7].y) > 0.05:
                cv2.imshow('RedPanda', img_head_tilt)
            elif landmarks[15].y - landmarks[7].y < -0.15 or landmarks[16].y - landmarks[8].y < -0.15:
                cv2.imshow('RedPanda', img_wave)
            else:
                cv2.imshow('RedPanda', img_default)

        cv2.imshow('RedPandaCaptured', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()