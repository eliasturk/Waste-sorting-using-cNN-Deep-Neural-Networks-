import cvzone
import cv2


cap = cv2.VideoCapture(0)


myClassifier = cvzone.Classifier('my_model','labels.txt')


fpsReader = cvzone.FPS()



while True :

    _, img =cap.read()

    predictions = myClassifier.getPrediction(img)

    fps, img = fpsReader.update(img, pos=(450, 100), color=(0, 0, 255), thickness=2)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
