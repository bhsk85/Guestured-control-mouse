import math
import cv2
import mediapipe as mp
import pyautogui as pag
import time

screen_width, screen_height = pag.size()

mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)


def main():
    ct = 0
    pt = 0
    dragging = False
    drag_start_pos = (0, 0)
    
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        vidh, vidw, vidz = img.shape

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = hands.process(imgRGB)
        ct = time.time()
        fps = int(round(1 / (ct - pt)))
        pt = ct
        cv2.putText(img, "FPS: {}".format(fps), (0, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        if data.multi_hand_landmarks:
            for hand_landmarks in data.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * vidw), int(landmark.y * vidh)
                    if landmark == hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]:
                        cv2.circle(img, (x, y), 10, (255, 255, 0), -1)
                    elif landmark == hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP]:
                        cv2.circle(img, (x, y), 10, (255, 0, 255), -1)
                
                index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP]
                middle_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]
                
                # Move the mouse cursor
                pag.moveTo(middle_finger_tip.x * screen_width * 1.3, middle_finger_tip.y * screen_height * 1.3)
                
                # Drag and drop functionality
                if not dragging and math.sqrt((thumb_tip.x * screen_width - index_finger_tip.x * screen_width) ** 2 +
                                             (thumb_tip.y * screen_height - index_finger_tip.y * screen_height) ** 2) < 50:
                    dragging = True
                    drag_start_pos = pag.position()
                
                if dragging and math.sqrt((thumb_tip.x * screen_width - index_finger_tip.x * screen_width) ** 2 +
                                          (thumb_tip.y * screen_height - index_finger_tip.y * screen_height) ** 2) >= 50:
                    dragging = False
                    drag_end_pos = pag.position()
                    pag.dragTo(drag_end_pos[0], drag_end_pos[1], duration=0.5)
                
                # Clicking functionality
                if math.sqrt((thumb_tip.x * screen_width - index_finger_tip.x * screen_width) ** 2 +
                             (thumb_tip.y * screen_height - index_finger_tip.y * screen_height) ** 2) < 50:
                    pag.click()
                    
        cv2.imshow('Gesture Controlled Mouse', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
