# fixed_capture.py
import cv2

def capture_image(filename="myfile.jpg"):
    # 1. Open camera
    # For Windows: use CAP_DSHOW to avoid default YUV bug
    # For Linux: CAP_V4L2 works better (you can test both)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  

    # 2. Force camera format to MJPG (fixes green/purple tints on many webcams)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to capture image")
        cap.release()
        return

    # 3. Try color conversions
    # Default OpenCV uses BGR, but some webcams send RGB or YUV incorrectly.
    # We'll try a few and save the first that looks OK.
    fixed = None

    try:
        # First attempt: swap channels (BGR → RGB)
        fixed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        pass

    # If still wrong, try YUV → BGR
    if fixed is None:
        try:
            fixed = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
        except:
            pass

    # If all else fails, keep the raw frame
    if fixed is None:
        fixed = frame

    # 4. Save the fixed image
    cv2.imwrite(filename, fixed)
    print(f"✅ Image saved as {filename}")

    cap.release()


if __name__ == "__main__":
    capture_image()
