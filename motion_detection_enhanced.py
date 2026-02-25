"""
Enhanced Motion Detection for AI Home Security System
Week 1 Sprint Demo - Professional Version
"""

import cv2
import datetime
import time
import numpy as np

class MotionDetector:
    def __init__(self, threshold=25, min_area=500, camera_id=0):
        """
        Initialize motion detector
        
        Args:
            threshold: Pixel difference threshold (0-255)
            min_area: Minimum contour area to consider as motion
            camera_id: Camera device ID
        """
        self.threshold = threshold
        self.min_area = min_area
        self.camera_id = camera_id
        self.motion_count = 0
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # Set camera properties (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Read first frames
        self.ret, self.frame1 = self.cap.read()
        self.ret, self.frame2 = self.cap.read()
        
        print("[SYSTEM] Motion Detector Initialized")
        print(f"[CONFIG] Threshold: {threshold}, Min Area: {min_area}")
        print("[STATUS] Ready. Press 'q' to quit, 's' to save snapshot\n")
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.fps = self.frame_count / elapsed_time
    
    def process_frame(self):
        """Process current frame and detect motion"""
        # Calculate difference
        diff = cv2.absdiff(self.frame1, self.frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, self.threshold, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        frame_copy = self.frame1.copy()
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_area:
                continue
            
            motion_detected = True
            self.motion_count += 1
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw rectangle
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add area text
            cv2.putText(frame_copy, f"Area: {int(area)}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Log to console (only print first few to avoid spam)
            if self.motion_count <= 5 or self.motion_count % 50 == 0:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] MOTION #{self.motion_count} - Area: {int(area)} at ({x}, {y})")
        
        return frame_copy, motion_detected
    
    def draw_overlay(self, frame, motion_detected):
        """Draw information overlay on frame"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Status
        status = "MOTION DETECTED!" if motion_detected else "Monitoring..."
        status_color = (0, 0, 255) if motion_detected else (0, 255, 0)
        
        # Draw semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (350, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Add text
        cv2.putText(frame, f"Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(frame, f"Time: {timestamp}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Motion Events: {self.motion_count}", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (250, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add threshold bar (visual indicator)
        bar_x, bar_y = 10, 110
        bar_w, bar_h = 200, 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        # Color-coded threshold level
        thresh_color = (0, 255, 0) if self.threshold < 30 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + int(self.threshold * 4), bar_y + bar_h), thresh_color, -1)
        cv2.putText(frame, f"Threshold: {self.threshold}", (bar_x + bar_w + 10, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def save_snapshot(self, frame):
        """Save current frame as snapshot"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[SAVED] Snapshot saved as: {filename}")
    
    def run(self):
        """Main loop"""
        print("[INFO] Motion detection running...")
        
        while True:
            # Calculate FPS
            self.calculate_fps()
            
            # Process frame
            processed_frame, motion_detected = self.process_frame()
            
            # Draw overlay
            display_frame = self.draw_overlay(processed_frame, motion_detected)
            
            # Show frame
            cv2.imshow("AI Home Security - Enhanced Motion Detection", display_frame)
            
            # Update frames
            self.frame1 = self.frame2
            self.ret, self.frame2 = self.cap.read()
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Stopping motion detection...")
                break
            elif key == ord('s'):
                self.save_snapshot(display_frame)
            elif key == ord('+') or key == ord('='):
                self.threshold = min(100, self.threshold + 5)
                print(f"[CONFIG] Threshold increased to: {self.threshold}")
            elif key == ord('-') or key == ord('_'):
                self.threshold = max(5, self.threshold - 5)
                print(f"[CONFIG] Threshold decreased to: {self.threshold}")
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"[SUMMARY] Total motion events detected: {self.motion_count}")
        print(f"[SUMMARY] Average FPS: {self.fps:.1f}")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    print("=" * 50)
    print("AI HOME SECURITY - MOTION DETECTION DEMO")
    print("=" * 50)
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save snapshot")
    print("  '+' - Increase sensitivity (higher threshold)")
    print("  '-' - Decrease sensitivity (lower threshold)")
    print("\nStarting in 3 seconds...\n")
    
    time.sleep(3)
    
    # Create and run detector
    detector = MotionDetector(
        threshold=25,    # Adjust for sensitivity
        min_area=500,    # Minimum area to consider as motion
        camera_id=0      # 0 = built-in webcam, 1 = USB camera
    )
    
    detector.run()