import cv2
import os

directory = 'Gestures/'
print(os.getcwd())

if not os.path.exists(directory):
    os.mkdir(directory)

if not os.path.exists(f'{directory}/blank'):
    os.mkdir(f'{directory}/blank')

labels = ['d', 'l', 'r', 'u', 'blank']
for letter in labels:
    if not os.path.exists(f'{directory}/{letter}'):
        os.mkdir(f'{directory}/{letter}')

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    
    count = {
        'd': len(os.listdir(directory + "/d")),
        'l': len(os.listdir(directory + "/l")),
        'r': len(os.listdir(directory + "/r")),
        'u': len(os.listdir(directory + "/u")),
        'blank': len(os.listdir(directory + "/blank"))
    }

    row = frame.shape[1]
    col = frame.shape[0]

    cv2.rectangle(frame, (0, 40), (300, 300), (255, 255, 255), 2)
    cv2.imshow("data", frame)

    frame = frame[40:300, 0:300]
    cv2.imshow("ROI", frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (48, 48))

    interrupt = cv2.waitKey(10)
    
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(f'{directory}/d/{count["d"]}.jpg', frame)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(f'{directory}/l/{count["l"]}.jpg', frame)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(f'{directory}/r/{count["r"]}.jpg', frame)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(f'{directory}/u/{count["u"]}.jpg', frame)
    if interrupt & 0xFF == ord('.'):
        cv2.imwrite(f'{directory}/blank/{count["blank"]}.jpg', frame)
    
    if interrupt & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
