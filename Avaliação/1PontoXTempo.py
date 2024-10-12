import cv2
import mediapipe as mp
import time



#cap = cv2.VideoCapture('videos/4.mp4')
# Video da webcam
video = cv2.VideoCapture(0)
# Gravar o video
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
gravacao = cv2.VideoWriter('Gravação.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
gravacao2 = cv2.VideoWriter('Gravação2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size) #video sem alterações

#Ferramentas
pose = mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils

#variaveis
pTime = 0
frames = 0

while video.isOpened():
    success, img = video.read()
    gravacao2.write(img)
    videoRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = Pose.process(videoRGB)
    points = results.pose_landmarks
    draw.draw_landmarks(img, points, pose.POSE_CONNECTIONS)
    h, w, _ = img.shape

    if points:

        # Write points into video
        gravacao.write(img)
        # Parar o video ao apertar a tecla
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        # Calcula frames
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(round(fps, 2)), (w - 50, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)











    cv2.imshow('Resultado', img)
    cv2.waitKey(1)

video.release()
gravacao.release()
gravacao2.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")