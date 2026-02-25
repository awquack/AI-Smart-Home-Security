"""
Clean Motion Detection - Optimized to reduce false boxes
"""

import cv2
import datetime
import time

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Allow camera to warm up
time.sleep(2)

# Read first frame
ret, frame1 = cap.read()
ret, frame2 = cap.read()

print("[STATUS] Motion detection started (CLEAN MODE)")
print("[INFO] Only significant motion will be shown")
print("Press 'q' to quit\n")

# Parameters for clean detection
MIN_AREA = 2000        # Much larger minimum area (was 500)
BLUR_SIZE = 21         # More blur to reduce noise
THRESHOLD_VALUE = 30   # Higher threshold for difference

while True:
    # Calculate difference
    diff = cv2.absdiff(frame1, frame2)
    
    # Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply HEAVY blur to remove noise
    blur = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    
    # Apply threshold
    _, thresh = cv2.threshold(blur, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    
    # Dilate to fill holes
    dilated = cv2.dilate(thresh, None, iterations=5)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank frame for display
    display_frame = frame2.copy()
    motion_detected = False
    box_count = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # IGNORE small areas (this is key!)
        if area < MIN_AREA:
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Additional filter: ignore very wide/tall boxes (often shadows)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 3 or aspect_ratio < 0.33:
            continue
        
        motion_detected = True
        box_count += 1
        
        # Draw ONLY ONE box around the whole motion area
        # Instead of drawing each contour, merge nearby boxes
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_frame, f"Motion {box_count}", (x, y-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw status
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = f"Motion: {box_count} area(s)" if motion_detected else "No Motion"
    color = (0, 0, 255) if motion_detected else (0, 255, 0)
    
    cv2.putText(display_frame, status, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(display_frame, timestamp, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show frame
    cv2.imshow("Clean Motion Detection", display_frame)
    
    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[STATUS] Motion detection stopped.")