from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
import math


class CricketBatSpeedAnalyzer:
    def __init__(self):
        self.model = YOLO('yolo11n-pose.pt')
        
        self.speed_history = deque(maxlen=15)  
        self.ema_speed = 0.0
        self.ema_alpha = 0.15 
        self.prev_wrist_left = None
        self.prev_wrist_right = None
        self.prev_timestamp = None
        
        self.prev_bat_angle = None
        self.angular_velocity_history = deque(maxlen=10)
        
        self.peak_speed = 0.0
        self.current_max_speed = 0.0
        
    def calculate_distance(self, point1, point2):
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def calculate_angle(self, point1, point2):
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        angle = math.degrees(math.atan2(dy, dx))
        return angle
    
    def calculate_bat_speed(self, keypoints, frame_time):
        
        if len(keypoints) < 17:
            return 0.0, 0.0, None, None
        
        try:
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]
            
            if (left_wrist[0] <= 0 or left_wrist[1] <= 0 or 
                right_wrist[0] <= 0 or right_wrist[1] <= 0):
                return 0.0, 0.0, None, None
            
            bat_angle = self.calculate_angle(left_wrist, right_wrist)
            
            angular_velocity = 0.0
            if self.prev_bat_angle is not None and frame_time > 0:
                angle_diff = bat_angle - self.prev_bat_angle
                
                if angle_diff > 180:
                    angle_diff -= 360
                elif angle_diff < -180:
                    angle_diff += 360
                
                angular_velocity = abs(angle_diff / frame_time)
            
            self.prev_bat_angle = bat_angle
            
            linear_speed = 0.0
            
            if self.prev_wrist_left is not None and self.prev_wrist_right is not None and frame_time > 0:
                left_dist = self.calculate_distance(self.prev_wrist_left, left_wrist)
                right_dist = self.calculate_distance(self.prev_wrist_right, right_wrist)

                avg_dist = (left_dist + right_dist) / 2.0
                linear_speed = avg_dist / frame_time 
            self.prev_wrist_left = left_wrist
            self.prev_wrist_right = right_wrist
            
            combined_speed = linear_speed
            
            return combined_speed, angular_velocity, left_wrist, right_wrist
            
        except Exception as e:
            return 0.0, 0.0, None, None
    
    def smooth_speed(self, speed):
        
        self.speed_history.append(speed)
        
        if len(self.speed_history) > 0:
            speed_ma = np.mean(list(self.speed_history))
        else:
            speed_ma = speed
        
        self.ema_speed = self.ema_alpha * speed_ma + (1 - self.ema_alpha) * self.ema_speed
        
        return self.ema_speed
    
    def speed_to_kmh(self, pixel_speed, pixels_per_meter=100):
    
        meters_per_second = pixel_speed / pixels_per_meter
        
        kmh = meters_per_second * 3.6
        
        return kmh
    
    def draw_skeleton_full_body(self, frame, keypoints):
        connections = [
            (5, 6),           # Shoulders
            (5, 7), (7, 9),   # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16)   # Right leg
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                
                if start_point[0] > 0 and start_point[1] > 0 and end_point[0] > 0 and end_point[1] > 0:
                    overlay = frame.copy()
                    cv2.line(overlay, 
                            (int(start_point[0]), int(start_point[1])), 
                            (int(end_point[0]), int(end_point[1])), 
                            (255, 255, 255), 6, cv2.LINE_AA)
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                    
                    cv2.line(frame, 
                            (int(start_point[0]), int(start_point[1])), 
                            (int(end_point[0]), int(end_point[1])), 
                            (255, 255, 255), 3, cv2.LINE_AA)
        
        for i in range(5, 17):
            if i < len(keypoints):
                point = keypoints[i]
                if point[0] > 0 and point[1] > 0:
                    overlay = frame.copy()
                    cv2.circle(overlay, (int(point[0]), int(point[1])), 8, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                    
                    cv2.circle(frame, (int(point[0]), int(point[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(frame, (int(point[0]), int(point[1])), 6, (200, 200, 200), 1, cv2.LINE_AA)
        
        return frame
    
    def draw_bat_line(self, frame, left_wrist, right_wrist):
        if left_wrist is not None and right_wrist is not None:
            overlay = frame.copy()
            cv2.line(overlay,
                    (int(left_wrist[0]), int(left_wrist[1])),
                    (int(right_wrist[0]), int(right_wrist[1])),
                    (100, 200, 255), 8, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            cv2.line(frame,
                    (int(left_wrist[0]), int(left_wrist[1])),
                    (int(right_wrist[0]), int(right_wrist[1])),
                    (50, 150, 255), 4, cv2.LINE_AA)
        
        return frame
    
    def draw_speed_ui(self, frame, speed_kmh, angular_velocity):
        """Dark theme UI matching weight transfer style"""
        h, w = frame.shape[:2]
        
        smooth_speed = self.smooth_speed(speed_kmh)
        
        if smooth_speed > self.current_max_speed:
            self.current_max_speed = smooth_speed
        
        panel_height = 70
        panel_y = h - panel_height
        
        overlay = frame.copy()
        for y in range(panel_y, h):
            alpha = 0.88 + 0.04 * ((y - panel_y) / panel_height)
            cv2.line(overlay, (0, y), (w, y), (8, 8, 10), 1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        
        title = "BAT SWING SPEED"
        title_font = cv2.FONT_HERSHEY_DUPLEX
        title_scale = 0.75
        title_thickness = 2
        
        title_x = 30
        title_y = panel_y + 28
        
        cv2.putText(frame, title, (title_x, title_y), 
                   title_font, title_scale, (200, 205, 210), title_thickness, cv2.LINE_AA)
        
        bar_start_x = int(w * 0.15)
        bar_width = int(w * 0.82)
        bar_y = panel_y + 42
        bar_height = 45
        
        max_speed = 180.0
        speed_ratio = min(smooth_speed / max_speed, 1.0)
        fill_width = int(bar_width * speed_ratio)
        
        cv2.rectangle(frame, (bar_start_x, bar_y), 
                     (bar_start_x + bar_width, bar_y + bar_height), 
                     (25, 25, 28), -1)
        
        if fill_width > 0:
            if speed_ratio < 0.4:  # Green zone
                color = (80, 180, 100)  # Muted green
            elif speed_ratio < 0.6:  # Yellow zone
                color = (100, 200, 200)  # Muted yellow
            elif speed_ratio < 0.8:  # Orange zone
                color = (100, 150, 220)  # Muted orange
            else:  # Red zone
                color = (100, 100, 240)  # Muted red
            
            cv2.rectangle(frame, (bar_start_x, bar_y), 
                         (bar_start_x + fill_width, bar_y + bar_height), 
                         color, -1)
        
        cv2.rectangle(frame, (bar_start_x, bar_y), 
                     (bar_start_x + bar_width, bar_y + bar_height), 
                     (180, 180, 180), 1, cv2.LINE_AA)
        
        speed_text = f"{int(smooth_speed)} km/h"
        speed_font = cv2.FONT_HERSHEY_DUPLEX
        speed_scale = 0.50
        speed_thickness = 1
        
        (speed_w, speed_h), _ = cv2.getTextSize(speed_text, speed_font, speed_scale, speed_thickness)
        speed_text_x = bar_start_x + (bar_width // 2) - (speed_w // 2)
        speed_text_y = bar_y + (bar_height + speed_h) // 2 - 1
        
        cv2.putText(frame, speed_text, (speed_text_x + 1, speed_text_y + 1), 
                   speed_font, speed_scale, (0, 0, 0), speed_thickness, cv2.LINE_AA)
        cv2.putText(frame, speed_text, (speed_text_x, speed_text_y), 
                   speed_font, speed_scale, (240, 240, 245), speed_thickness, cv2.LINE_AA)
        
        label_font = cv2.FONT_HERSHEY_DUPLEX
        label_scale = 0.35
        label_thickness = 1
        label_y = panel_y + 62
        
        peak_text = f"PEAK: {int(self.current_max_speed)} km/h"
        peak_label_x = bar_start_x
        cv2.putText(frame, peak_text, (peak_label_x, label_y), 
                   label_font, label_scale, (120, 200, 150), label_thickness, cv2.LINE_AA)
        
        ang_vel_text = f"ROTATION: {int(angular_velocity)}°/s"
        (ang_w, _), _ = cv2.getTextSize(ang_vel_text, label_font, label_scale, label_thickness)
        ang_label_x = bar_start_x + bar_width - ang_w
        cv2.putText(frame, ang_vel_text, (ang_label_x, label_y), 
                   label_font, label_scale, (150, 150, 200), label_thickness, cv2.LINE_AA)
        
        return frame
    
    def process_video(self, video_path, output_path='cricket_bat_speed.mp4', 
                     slow_motion=False, pixels_per_meter=100):
       
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_fps = fps if not slow_motion else fps // 2
        frame_time = 1.0 / fps  
        
        print(f"🏏 Processing Cricket Bat Swing Speed - Ultra Smooth")
        print(f"Resolution: {width}x{height} @ {fps}fps -> Output @ {output_fps}fps")
        print(f"Total frames: {total_frames}")
        print(f"Smoothing: 15-frame moving average + EMA (alpha=0.15)")
        print(f"Calibration: {pixels_per_meter} pixels = 1 meter")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model(frame, verbose=False)
            
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy[0].cpu().numpy()
                
                frame = self.draw_skeleton_full_body(frame, keypoints)
                
                combined_speed, angular_velocity, left_wrist, right_wrist = \
                    self.calculate_bat_speed(keypoints, frame_time)
                
                speed_kmh = self.speed_to_kmh(combined_speed, pixels_per_meter)
                
                frame = self.draw_speed_ui(frame, speed_kmh, angular_velocity)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Progress: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")
        
        cap.release()
        out.release()
        print(f"Done! Peak speed recorded: {int(self.current_max_speed)} km/h")
        print(f"Output saved: {output_path}")


analyzer = CricketBatSpeedAnalyzer()

analyzer.process_video("videos/ishan_kishan.mp4", "videos/bat_swing_speed_ishan_kishan.mp4", 
                      slow_motion=False, pixels_per_meter=50)

# OR slow motion
# analyzer.process_video("cricket_batting.mp4", "cricket_bat_speed_slow.mp4", 
#                        slow_motion=True, pixels_per_meter=100)