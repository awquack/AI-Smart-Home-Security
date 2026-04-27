import cv2

# Try different camera indexes
for index in range(3):
    cap = cv2.VideoCapture(index)

    if cap.isOpened():
        print("Camera found at index", index)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        break

    cap.release()

else:
    print("No camera detected")