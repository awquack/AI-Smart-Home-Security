"""
AI Home Security - Sprint 1 Demo
FIXED: Boxes stay on screen so audience can see them
"""

import cv2
import datetime
import time
import os
from collections import deque

# Create snapshots directory
if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

# Initialize camera
print("[INIT] Starting camera...")
cap = cv2.VideoCapture(0)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Allow camera to warm up
time.sleep(2)

# Read first frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Parameters
MIN_AREA = 2500
BLUR_SIZE = 15
THRESHOLD = 25

# FIX: Store recent detections with timestamps
# Each detection will stay on screen for 1 second
recent_detections = deque(maxlen=10)  # Store up to 10 recent boxes

print("=" * 60)
print("AI HOME SECURITY - SPRINT 1 DEMO")
print("=" * 60)
print("\n🎥 Camera: READY")
print("⚙️  Mode: Persistent Motion Boxes (boxes stay 1 second)")
print(f"📏 Min Area: {MIN_AREA} pixels")
print("\n📋 Controls:")
print("   'q' - Quit demo")
print("   's' - Save snapshot")
print("=" * 60)
print("\n[STATUS] Monitoring...\n")

# Tracking variables
motion_count = 0
frame_count = 0
start_time = time.time()

while True:
    # Calculate FPS
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    
    # Calculate difference
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    _, thresh = cv2.threshold(blur, THRESHOLD, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=5)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get current time
    current_time = time.time()
    
    # Process new detections
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < MIN_AREA:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Aspect ratio filter
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio > 4 or aspect_ratio < 0.25:
            continue
        
        # FIX: Store this detection with timestamp
        # It will be displayed for 1 second
        detection = {
            'box': (x, y, w, h),
            'time': current_time,
            'id': motion_count
        }
        recent_detections.append(detection)
        motion_count += 1
        
        # Log to console
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 🔴 MOTION #{motion_count} - Area: {int(area)}")
        break  # Only take the largest contour per frame
    
    # Create display frame
    display_frame = frame2.copy()
    
    # FIX: Draw ALL recent detections that are less than 1 second old
    active_detections = []
    for det in recent_detections:
        age = current_time - det['time']
        if age < 1.0:  # Show for 1 second
            active_detections.append(det)
            x, y, w, h = det['box']
            
            # Calculate opacity based on age (fade out effect)
            opacity = max(0.3, 1.0 - age)  # Fade from 1.0 to 0.3
            
            # Draw thick box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Draw filled top bar for better visibility
            cv2.rectangle(display_frame, (x, y-25), (x + 150, y), (0, 0, 255), -1)
            
            # Add text
            cv2.putText(display_frame, f"INTRUSION #{det['id']}", (x+5, y-8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw status overlay - LARGE and CLEAR
    # Semi-transparent black background
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 120), (0, 0, 0), -1)
    display_frame = cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0)
    
    # BIG status text
    if active_detections:
        status_text = f"🔴 INTRUSION DETECTED! ({len(active_detections)})"
        status_color = (0, 0, 255)
    else:
        status_text = "🟢 MONITORING - NO INTRUSION"
        status_color = (0, 255, 0)
    
    cv2.putText(display_frame, status_text, (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
    
    # Timestamp and info
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(display_frame, f"📅 {timestamp}", (20, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(display_frame, f"📊 Total Events: {motion_count} | FPS: {fps:.1f}", (20, 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Show frame
    cv2.imshow("AI Home Security - Sprint 1 Demo (PERSISTENT BOXES)", display_frame)
    
    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()
    
    # Handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        snapshot_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshots/event_{snapshot_time}.jpg"
        cv2.imwrite(filename, display_frame)
        print(f"📸 Snapshot saved: {filename}")

# Clean up
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("DEMO COMPLETED")
print(f"📊 Total motion events: {motion_count}")
print("=" * 60)