import cv2


class Camera:

    def __init__(self):

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            raise RuntimeError("Could not open webcam")

        print("Camera initialized successfully!")


    def get_frame(self):

        success, frame = self.camera.read()

        if not success:
            return None

        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not success:
            return None

        return buffer.tobytes()


    def release(self):

        if self.camera.isOpened():
            self.camera.release()

            print("Camera released.")