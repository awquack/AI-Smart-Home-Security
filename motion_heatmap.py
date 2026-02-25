"""
Merged Boxes - Combines nearby boxes into single detection
"""

import cv2
import numpy as np
import datetime
import time

def merge_nearby_boxes(contours, merge_distance=100):
    """
    Merge contours that are close to each other
    """
    if not contours:
        return []
    
    # Get bounding boxes for all contours
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x+w, y+h])  # [x1, y1, x2, y2]
    
    boxes = np.array(boxes)
    
    # Simple merging: if boxes overlap or are close, merge them
    merged = []
    used = set()
    
    for i in range(len(boxes)):
        if i in used:
            continue
        
        current_box = boxes[i].copy()
        used.add(i)
        
        # Check other boxes
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            
            # Calculate distance between boxes
            box1_center = [(boxes[i][0] + boxes[i][2])/2, (boxes[i][1] + boxes[i][3])/2]
            box2_center = [(boxes[j][0] + boxes[j][2])/2, (boxes[j][1] + boxes[j][3])/2]
            
            distance = np.sqrt((box1_center[0] - box2_center[0])**2 + 
                              (box1_center[1] - box2_center[1])**2)
            
            # If boxes are close, merge them
            if distance < merge_distance:
                current_box[0] = min(current_box[0], boxes[j][0])
                current_box[1] = min(current_box[1], boxes[j][1])
                current_box[2] = max(current_box[2], boxes[j][2])
                current_box[3] = max(current_box[3], boxes[j][3])
                used.add(j)
        
        merged.append(current_box)
    
    return merged

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Allow camera to warm up
time.sleep(2)

# Read first frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

print("[STATUS] Merged Box Detection Started")
print("[INFO] Nearby boxes will be combined into one")
print("Press 'q' to quit\n")

while True:
    # Calculate difference
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours
    valid_contours = [c for c in contours if cv2.contourArea(c) > 300]
    
    # MERGE nearby boxes
    merged_boxes = merge_nearby_boxes(valid_contours, merge_distance=150)
    
    # Draw on frame
    display_frame = frame2.copy()
    
    for box in merged_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(display_frame, "INTRUSION", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show original contours in small window (for comparison)
    contour_frame = np.zeros_like(frame1)
    cv2.drawContours(contour_frame, valid_contours, -1, (0, 255, 0), 1)
    
    # Status text
    status = f"INTRUSION DETECTED ({len(merged_boxes)} zone(s))" if merged_boxes else "MONITORING"
    color = (0, 0, 255) if merged_boxes else (0, 255, 0)
    
    cv2.putText(display_frame, status, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(display_frame, f"Merged Boxes: {len(merged_boxes)}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Show frames
    cv2.imshow("Merged Detection (Clean)", display_frame)
    cv2.imshow("Raw Contours (Before Merge)", contour_frame)
    
    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[STATUS] Stopped.")