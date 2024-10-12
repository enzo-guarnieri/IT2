import cv2
import mediapipe as mp
import math

# Video já feito
#video = cv2.VideoCapture('polichinelos.mp4')

# Video da webcam
video = cv2.VideoCapture(0)


pose = mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils
contador = 0
check = True

while True:
    success, img = video.read()
    videoRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Pose.process(videoRGB)
    points = results.pose_landmarks
    draw.draw_landmarks(img, points, pose.POSE_CONNECTIONS)

    h, w, _ = img.shape

    if points:
        # Pontos relevantes do pe, ja convertidos em pixels
        peDY = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].y * h)
        peDX = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].x * w)
        peEY = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].y * h)
        peEX = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].x * w)
        # Pontos relevantes da mao, ja convertidos em pixels
        moDY = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].y * h)
        moDX = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].x * w)
        moEY = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].y * h)
        moEX = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].x * w)

        distMO = math.hypot(moDX - moEX, moDY - moEY)
        distPE = math.hypot(peDX - peEX, peDY - peEY)
        print(f"maos {distMO}, pes {distPE}")

        if check and distMO <= 150 and distPE >= 150:
            contador += 1
            check = False

        if distMO > 150 and distPE < 150:
            check = True

        texto = f'Qnt: {contador}'
        cv2.putText(img, texto, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)

    cv2.imshow('Resultado', img)
    cv2.waitKey(40)
