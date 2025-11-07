from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque


class CricketWeightTransferAnalyzer:
    def __init__(self):
        self.model = YOLO('yolo11n-pose.pt')
        self.weight_history = deque(maxlen=5)  

        self.ema_front = 50.0
        self.ema_back = 50.0
        self.ema_alpha = 0.15
        
    def calculate_front_back_weight(self, keypoints):
        if len(keypoints) < 17:
            return 50.0, 50.0, None, None
        
        try:
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_elbow = keypoints[7]
            right_elbow = keypoints[8]
            left_wrist = keypoints[9]
            right_wrist = keypoints[10]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            left_knee = keypoints[13]
            right_knee = keypoints[14]
            left_ankle = keypoints[15]
            right_ankle = keypoints[16]
            
            com_x = (
                (left_shoulder[0] + right_shoulder[0]) * 0.075 +
                (left_elbow[0] + right_elbow[0]) * 0.05 +
                (left_wrist[0] + right_wrist[0]) * 0.025 +
                (left_hip[0] + right_hip[0]) * 0.20 +
                (left_knee[0] + right_knee[0]) * 0.10 +
                (left_ankle[0] + right_ankle[0]) * 0.05
            )
            
            com_y = (
                (left_shoulder[1] + right_shoulder[1]) * 0.075 +
                (left_elbow[1] + right_elbow[1]) * 0.05 +
                (left_wrist[1] + right_wrist[1]) * 0.025 +
                (left_hip[1] + right_hip[1]) * 0.20 +
                (left_knee[1] + right_knee[1]) * 0.10 +
                (left_ankle[1] + right_ankle[1]) * 0.05
            )
            
            ankle_x_diff = abs(left_ankle[0] - right_ankle[0])
            ankle_y_diff = abs(left_ankle[1] - right_ankle[1])
            
            if ankle_x_diff > ankle_y_diff:
                if left_ankle[0] > right_ankle[0]:
                    front_foot = left_ankle
                    back_foot = right_ankle
                else:
                    front_foot = right_ankle
                    back_foot = left_ankle
                
                front_distance = abs(com_x - front_foot[0])
                back_distance = abs(com_x - back_foot[0])
            else:
                if left_ankle[1] < right_ankle[1]:
                    front_foot = left_ankle
                    back_foot = right_ankle
                else:
                    front_foot = right_ankle
                    back_foot = left_ankle
                
                front_distance = abs(com_y - front_foot[1])
                back_distance = abs(com_y - back_foot[1])
            
            total_distance = front_distance + back_distance
            
            if total_distance > 0:
                back_weight = (front_distance / total_distance) * 100
                front_weight = 100 - back_weight
            else:
                front_weight = 50.0
                back_weight = 50.0
            
            com_point = (int(com_x), int(com_y))
            
            return front_weight, back_weight, com_point, (front_foot, back_foot)
            
        except:
            return 50.0, 50.0, None, None
    
    def smooth_weights(self, front_weight, back_weight):        
        self.weight_history.append((front_weight, back_weight))
        
        if len(self.weight_history) > 0:
            front_ma = np.mean([wt[0] for wt in self.weight_history])
            back_ma = np.mean([wt[1] for wt in self.weight_history])
        else:
            front_ma = front_weight
            back_ma = back_weight
        
        # EMA formula: new_value = alpha * new_data + (1 - alpha) * old_ema
        self.ema_front = self.ema_alpha * front_ma + (1 - self.ema_alpha) * self.ema_front
        self.ema_back = self.ema_alpha * back_ma + (1 - self.ema_alpha) * self.ema_back
        
        return self.ema_front, self.ema_back
    
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
    
    def draw_fixed_center_bar_ui(self, frame, front_weight, back_weight):
        """Dark theme UI with FIXED center line and smooth transitions"""
        h, w = frame.shape[:2]
        
        front_avg, back_avg = self.smooth_weights(front_weight, back_weight)
        
        panel_height = 70
        panel_y = h - panel_height
        
        overlay = frame.copy()
        for y in range(panel_y, h):
            alpha = 0.88 + 0.04 * ((y - panel_y) / panel_height)
            cv2.line(overlay, (0, y), (w, y), (8, 8, 10), 1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        
        title = "WEIGHT TRANSFER"
        title_font = cv2.FONT_HERSHEY_DUPLEX
        title_scale = 0.42
        title_thickness = 1
        
        title_x = 20
        title_y = panel_y + 16
        
        cv2.putText(frame, title, (title_x, title_y), 
                   title_font, title_scale, (160, 165, 170), title_thickness, cv2.LINE_AA)
        
        bar_start_x = int(w * 0.18)
        bar_width = int(w * 0.78)
        bar_y = panel_y + 22
        bar_height = 30
        
        center_x = bar_start_x + (bar_width // 2)
        max_half_width = bar_width // 2
        
        back_fill_ratio = (back_avg / 100.0)
        back_fill_width = int(max_half_width * back_fill_ratio)
        back_start_x = center_x - back_fill_width
        
        front_fill_ratio = (front_avg / 100.0)
        front_fill_width = int(max_half_width * front_fill_ratio)
        front_end_x = center_x + front_fill_width
        
        cv2.rectangle(frame, (bar_start_x, bar_y), 
                     (bar_start_x + bar_width, bar_y + bar_height), 
                     (25, 25, 28), -1)
        
        if back_fill_width > 0:
            cv2.rectangle(frame, (back_start_x, bar_y), 
                         (center_x, bar_y + bar_height), 
                         (35, 45, 140), -1)
        
        if front_fill_width > 0:
            cv2.rectangle(frame, (center_x, bar_y), 
                         (front_end_x, bar_y + bar_height), 
                         (140, 100, 50), -1)
        
        cv2.rectangle(frame, (bar_start_x, bar_y), 
                     (bar_start_x + bar_width, bar_y + bar_height), 
                     (180, 180, 180), 1, cv2.LINE_AA)
        
        cv2.line(frame, (center_x, bar_y - 2), (center_x, bar_y + bar_height + 2), 
                (240, 240, 240), 2, cv2.LINE_AA)
        
        center_text = "50%"
        center_font = cv2.FONT_HERSHEY_DUPLEX
        center_scale = 0.32
        center_thickness = 1
        (center_w, _), _ = cv2.getTextSize(center_text, center_font, center_scale, center_thickness)
        cv2.putText(frame, center_text, (center_x - center_w // 2, bar_y - 5), 
                   center_font, center_scale, (160, 165, 170), center_thickness, cv2.LINE_AA)
        
        percent_font = cv2.FONT_HERSHEY_DUPLEX
        percent_scale = 0.48
        percent_thickness = 1
        
        back_text = f"{int(back_avg)}%"
        front_text = f"{int(front_avg)}%"
        
        if back_fill_width > 50:
            (back_w, back_h), _ = cv2.getTextSize(back_text, percent_font, percent_scale, percent_thickness)
            back_text_x = back_start_x + (back_fill_width // 2) - (back_w // 2)
            back_text_y = bar_y + (bar_height + back_h) // 2 - 1
            
            cv2.putText(frame, back_text, (back_text_x + 1, back_text_y + 1), 
                       percent_font, percent_scale, (0, 0, 0), percent_thickness, cv2.LINE_AA)
            cv2.putText(frame, back_text, (back_text_x, back_text_y), 
                       percent_font, percent_scale, (240, 240, 245), percent_thickness, cv2.LINE_AA)
        
        if front_fill_width > 50:
            (front_w, front_h), _ = cv2.getTextSize(front_text, percent_font, percent_scale, percent_thickness)
            front_text_x = center_x + (front_fill_width // 2) - (front_w // 2)
            front_text_y = bar_y + (bar_height + front_h) // 2 - 1
            
            cv2.putText(frame, front_text, (front_text_x + 1, front_text_y + 1), 
                       percent_font, percent_scale, (0, 0, 0), percent_thickness, cv2.LINE_AA)
            cv2.putText(frame, front_text, (front_text_x, front_text_y), 
                       percent_font, percent_scale, (240, 240, 245), percent_thickness, cv2.LINE_AA)
        
        label_font = cv2.FONT_HERSHEY_DUPLEX
        label_scale = 0.35
        label_thickness = 1
        label_y = panel_y + 62
        
        back_label_x = bar_start_x + (max_half_width // 2)
        (back_label_w, _), _ = cv2.getTextSize("BACK", label_font, label_scale, label_thickness)
        cv2.putText(frame, "BACK", (back_label_x - back_label_w // 2, label_y), 
                   label_font, label_scale, (80, 85, 150), label_thickness, cv2.LINE_AA)
        
        forward_label_x = center_x + (max_half_width // 2)
        (forward_label_w, _), _ = cv2.getTextSize("FORWARD", label_font, label_scale, label_thickness)
        cv2.putText(frame, "FORWARD", (forward_label_x - forward_label_w // 2, label_y), 
                   label_font, label_scale, (150, 120, 80), label_thickness, cv2.LINE_AA)
        
        return frame
    
    def process_video(self, video_path, output_path='cricket_smooth.mp4', slow_motion=False):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_fps = fps if not slow_motion else fps // 2
        
        print(f"🏏 Processing Cricket Weight Transfer - Ultra Smooth")
        print(f"Resolution: {width}x{height} @ {fps}fps -> Output @ {output_fps}fps")
        print(f"Total frames: {total_frames}")
        print(f"Smoothing: 15-frame moving average + EMA (alpha=0.15)")
        
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
                
                front_weight, back_weight, com_point, feet_points = self.calculate_front_back_weight(keypoints)
                
                frame = self.draw_fixed_center_bar_ui(frame, front_weight, back_weight)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Progress: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")
        
        cap.release()
        out.release()
        print(f"Done! output saved: {output_path}")


analyzer = CricketWeightTransferAnalyzer()

analyzer.process_video("videos/virat_kohli.mp4", "videos/virat_kohli_weight_transfer.mp4", slow_motion=False)

# OR slow motion
# analyzer.process_video("cricket_batting.mp4", "cricket_smooth_slow.mp4", slow_motion=True)
