import cv2
import mediapipe as mp



# Video da webcam
video = cv2.VideoCapture(0)
# Gravar o video
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
gravacao = cv2.VideoWriter('Gravação.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
gravacao2 = cv2.VideoWriter('Gravação2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

pose = mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils
contador = 0
check = True


while True:
    success, img = video.read()
    gravacao2.write(img)
    videoRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Pose.process(videoRGB)
    points = results.pose_landmarks
    draw.draw_landmarks(img, points, pose.POSE_CONNECTIONS)
    h, w, _ = img.shape

    if points:
        # Pontos relevantes da mao
        moDY = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].y*h)
        moDX = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].x*w)
        moEY = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].y * h)
        moEX = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].x * w)
        # Pontos relevantes do ombro
        obDY = int(points.landmark[pose.PoseLandmark.RIGHT_SHOULDER].y*h)
        obDX = int(points.landmark[pose.PoseLandmark.RIGHT_SHOULDER].x * w)
        obEY = int(points.landmark[pose.PoseLandmark.LEFT_SHOULDER].y * h)
        obEX = int(points.landmark[pose.PoseLandmark.LEFT_SHOULDER].x * w)
        # Pontos relevantes do cotovelo
        ctDY = int(points.landmark[pose.PoseLandmark.RIGHT_ELBOW].y * h)
        ctDX = int(points.landmark[pose.PoseLandmark.RIGHT_ELBOW].x * w)
        ctEY = int(points.landmark[pose.PoseLandmark.LEFT_ELBOW].y * h)
        ctEX = int(points.landmark[pose.PoseLandmark.LEFT_ELBOW].x * w)

        if check and moDY <= obDY and ctDY <= obDY and moEY <= obEY and ctEY <= obEY:
            contador += 1
            check = False

        if moDY > obDY and ctDY > obDY and moEY > obEY and ctEY > obEY:
            check = True

        texto = f'Qnt: {contador}'
        cv2.putText(img, texto, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
        gravacao.write(img)
        # Parar o video ao apertar a tecla s
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cv2.imshow('Resultado', img)

    cv2.waitKey(1)

video.release()
gravacao.release()
gravacao2.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")
