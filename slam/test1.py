import cv2
import torch
import math
import numpy as np
import time
import os
import pickle
import requests
from collections import defaultdict, deque

# =============== Load Scaled Depth Factor ===============
try:
    DEPTH_SCALE = np.load("depth_scale_factor.npy").item()
except Exception:
    DEPTH_SCALE = 5.0

# =============== MiDaS Depth Model on GPU/CPU ===============
print("🔧 Initializing MiDaS...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# CPU optimizations
if not torch.cuda.is_available():
    print("⚠️  CPU MODE - Applying optimizations...")
    torch.set_num_threads(4)
else:
    print(f"✅ CUDA - GPU: {torch.cuda.get_device_name(0)}")


class RoverStreamCapture:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.resp = None
        self.bytes_buf = b""
        self.is_opened = False
        self._connect()

    def _connect(self):
        try:
            self.resp = requests.get(self.stream_url, stream=True, timeout=5)
            self.is_opened = (self.resp.status_code == 200)
        except Exception:
            self.is_opened = False

    def isOpened(self):
        return self.is_opened

    def read(self):
        if not self.is_opened:
            return False, None
        try:
            for chunk in self.resp.iter_content(chunk_size=1024):
                self.bytes_buf += chunk
                a = self.bytes_buf.find(b'\xff\xd8')
                b = self.bytes_buf.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = self.bytes_buf[a:b+2]
                    self.bytes_buf = self.bytes_buf[b+2:]
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        return True, img
            return False, None
        except Exception:
            self.is_opened = False
            return False, None

    def release(self):
        if self.resp is not None:
            self.resp.close()
        self.is_opened = False


class FastSLAM:
    """Fast SLAM with dynamic feature display and accurate pose estimation"""
    
    def __init__(self, fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                 lookahead_height_ratio=0.9,
                 min_features_per_frame=50, max_features_per_frame=120,
                 rotation_sensitivity=3.0, feature_lifetime_frames=30,
                 max_depth_threshold=4.5, spatial_grid_size=0.10,
                 depth_compute_interval=2, max_display_features=200):
        
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        
        self.pose = np.zeros(3, dtype=float)
        self.trajectory = []
        self.path_distance = 0.0
        self.meter_markers = []
        
        self.motion_mode = "REST"
        self.FIXED_FORWARD_DISTANCE = 0.004
        self.ROTATION_SENSITIVITY = rotation_sensitivity
        self.rest_frames_count = 0
        self.rest_frames_threshold = 10
        
        # Feature limits
        self.min_features_per_frame = min_features_per_frame
        self.max_features_per_frame = max_features_per_frame
        self.current_feature_target = max_features_per_frame
        
        # Spatial grid for ALL features (stored)
        self.spatial_grid_size = spatial_grid_size
        self.spatial_grid = {}
        self.max_map_features = 15000
        
        # NEW: Display only recent features for speed
        self.max_display_features = max_display_features
        self.recent_features = deque(maxlen=max_display_features)  # Only recent for display
        
        self.feature_lifetime_frames = feature_lifetime_frames
        
        self.max_depth_threshold = max_depth_threshold
        
        # Depth optimization
        self.depth_compute_interval = depth_compute_interval
        self.frame_skip_counter = 0
        self.last_depth_map = None
        
        # Loop closure
        self.loop_closure_history = []
        self.loop_closure_threshold = 0.90
        self.loop_closure_cooldown = 0
        self.loop_closure_cooldown_frames = 100
        
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Optimized ORB
        self.orb = cv2.ORB_create(
            nfeatures=250,
            scaleFactor=1.2,
            nlevels=6,
            edgeThreshold=10,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=10
        )
        
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.lookahead_height_ratio = lookahead_height_ratio
        self.lookahead_pixel = None
        self.lookahead_depth_curr = None
        self.lookahead_world_pos = None
        
        self.prev_gray = None
        self.prev_depth = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_tracked_features = None
        
        self.frame_count = 0
        
        self.current_features = []
        self.feature_counts = {'total': 0, 'valid_depth': 0, 'displayed': 0, 'stored': 0}
        self.rotation_magnitude = 0.0
        self.translation_magnitude = 0.0
        
        self.last_map_filename = None
        self.last_map_pkl = None
        
        self.loop_closure_detected = False

    def _preprocess_frame(self, gray):
        enhanced = self.clahe.apply(gray)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=5)
        return enhanced

    def _spatial_grid_key(self, x, y):
        return (int(x / self.spatial_grid_size), int(y / self.spatial_grid_size))

    def _add_to_spatial_grid(self, x, y, z, descriptor, quality, frame, cam_x, cam_y):
        key = self._spatial_grid_key(x, y)
        cam_distance = math.sqrt((x - cam_x)**2 + (y - cam_y)**2)
        
        if key in self.spatial_grid:
            existing = self.spatial_grid[key]
            if quality > existing['quality']:
                self.spatial_grid[key] = {
                    'x': x, 'y': y, 'z': z,
                    'descriptor': descriptor,
                    'quality': quality,
                    'frame': frame,
                    'cam_distance': cam_distance
                }
        else:
            if len(self.spatial_grid) < self.max_map_features:
                self.spatial_grid[key] = {
                    'x': x, 'y': y, 'z': z,
                    'descriptor': descriptor,
                    'quality': quality,
                    'frame': frame,
                    'cam_distance': cam_distance
                }

    def _compute_lookahead_pixel(self, img_height, img_width):
        u = self.cx
        v = img_height * self.lookahead_height_ratio
        return (int(u), int(v))

    def _midas_to_metric_depth(self, midas_value):
        if midas_value < 1e-3:
            return None
        return DEPTH_SCALE / (midas_value + 1e-6)

    def _backproject_to_world(self, u, v, metric_depth, pose):
        if metric_depth is None or metric_depth <= 0:
            return None
        
        X_cam = (u - self.cx) * metric_depth / self.fx
        Y_cam = (v - self.cy) * metric_depth / self.fy
        Z_cam = metric_depth
        
        yaw = pose[2]
        c, s = math.cos(yaw), math.sin(yaw)
        x_world = pose[0] + (c * Z_cam + s * X_cam)
        y_world = pose[1] + (s * Z_cam - c * X_cam)
        
        return (x_world, y_world, Z_cam)

    def _extract_depth_pattern(self, depth_map, keypoints):
        if len(keypoints) < 10:
            return None
        
        depths = []
        for kp in keypoints[:20]:
            u, v = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                midas_depth = depth_map[v, u]
                metric_depth = self._midas_to_metric_depth(midas_depth)
                if metric_depth is not None and metric_depth <= self.max_depth_threshold:
                    depths.append(metric_depth)
        
        if len(depths) < 5:
            return None
        
        depths = np.array(depths)
        pattern = depths / (np.max(depths) + 1e-6)
        return pattern

    def _check_loop_closure(self, current_pattern):
        if self.loop_closure_cooldown > 0:
            self.loop_closure_cooldown -= 1
            return False
        
        if current_pattern is None or len(self.loop_closure_history) < 5:
            return False
        
        min_frame_gap = 100
        current_frame = self.frame_count
        
        for past_frame, past_pattern in self.loop_closure_history[-20:]:
            if current_frame - past_frame < min_frame_gap:
                continue
            
            if len(current_pattern) == len(past_pattern):
                similarity = np.corrcoef(current_pattern, past_pattern)[0, 1]
                if similarity > self.loop_closure_threshold:
                    self.loop_closure_cooldown = self.loop_closure_cooldown_frames
                    return True
        
        return False

    def _adjust_feature_target(self):
        if len(self.current_features) < self.min_features_per_frame:
            self.current_feature_target = min(
                self.current_feature_target + 10,
                self.max_features_per_frame
            )
        elif len(self.current_features) > self.max_features_per_frame:
            self.current_feature_target = max(
                self.current_feature_target - 10,
                self.min_features_per_frame
            )

    def _detect_and_build_map(self, gray, depth_map, pose):
        enhanced = self._preprocess_frame(gray)
        img_height, img_width = gray.shape
        
        mask = np.zeros_like(enhanced, dtype=np.uint8)
        h_start = int(img_height * 0.33)
        mask[h_start:, :] = 255
        
        kp_orb, desc_orb = self.orb.detectAndCompute(enhanced, mask=mask)
        
        if kp_orb is None or desc_orb is None or len(kp_orb) == 0:
            self.feature_counts['total'] = 0
            self.feature_counts['valid_depth'] = 0
            return []
        
        kp_desc_pairs = list(zip(kp_orb, desc_orb))
        kp_desc_pairs.sort(key=lambda x: -x[0].response)
        
        n_features = min(len(kp_desc_pairs), self.max_features_per_frame)
        kp_desc_pairs = kp_desc_pairs[:n_features]
        
        kp_orb = [kp for kp, _ in kp_desc_pairs]
        desc_orb = np.array([desc for _, desc in kp_desc_pairs])
        
        # Loop closure (every 5 frames)
        if self.frame_count % 5 == 0:
            depth_pattern = self._extract_depth_pattern(depth_map, kp_orb)
            
            if self._check_loop_closure(depth_pattern):
                if not self.loop_closure_detected:
                    print("🔄 LOOP CLOSURE")
                    self.loop_closure_detected = True
            else:
                self.loop_closure_detected = False
            
            if depth_pattern is not None:
                self.loop_closure_history.append((self.frame_count, depth_pattern))
                if len(self.loop_closure_history) > 50:
                    self.loop_closure_history.pop(0)
        
        valid_depth_count = 0
        new_features = []
        all_features = []
        
        cam_x, cam_y = pose[0], pose[1]
        
        for i, kp in enumerate(kp_orb):
            u, v = int(kp.pt[0]), int(kp.pt[1])
            if not (0 <= u < img_width and 0 <= v < img_height):
                continue
            
            midas_depth = depth_map[v, u]
            metric_depth = self._midas_to_metric_depth(midas_depth)
            
            if metric_depth is not None and metric_depth <= self.max_depth_threshold:
                valid_depth_count += 1
                world_pos = self._backproject_to_world(u, v, metric_depth, pose)
                
                if world_pos is not None:
                    x_world, y_world, z_cam = world_pos
                    descriptor = desc_orb[i]
                    quality = kp.response
                    
                    # Add to full map (stored)
                    self._add_to_spatial_grid(
                        x_world, y_world, metric_depth, 
                        descriptor, quality, self.frame_count,
                        cam_x, cam_y
                    )
                    
                    # Add to recent display buffer (fast rendering)
                    self.recent_features.append({
                        'x': x_world,
                        'y': y_world,
                        'z': metric_depth,
                        'u': u,
                        'v': v,
                        'frame': self.frame_count
                    })
                    
                    new_features.append((u, v))
                    all_features.append((u, v))
        
        self.feature_counts['total'] = len(kp_orb)
        self.feature_counts['valid_depth'] = valid_depth_count
        self.feature_counts['stored'] = len(self.spatial_grid)
        self.feature_counts['displayed'] = len(self.recent_features)
        
        self.current_features = all_features
        
        self._adjust_feature_target()
        
        self.prev_keypoints = kp_orb
        self.prev_descriptors = desc_orb
        
        return new_features

    def _detect_motion_state(self, gray):
        if self.prev_gray is None or self.prev_tracked_features is None:
            return False, False
        
        prev_pts = np.array(self.prev_tracked_features, dtype=np.float32).reshape(-1, 1, 2)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )
        
        if status is None or np.sum(status) < 5:
            return False, False
        
        good_prev = prev_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]
        
        displacements_h = good_next[:, 0, 0] - good_prev[:, 0, 0]
        displacements_v = good_next[:, 0, 1] - good_prev[:, 0, 1]
        
        median_h = np.median(displacements_h)
        median_v = np.median(displacements_v)
        
        has_rotation = abs(median_h) > 0.5
        has_forward_motion = median_v > 0.05
        
        return has_forward_motion, has_rotation

    def depth_map(self, frame_bgr):
        self.frame_skip_counter += 1
        if self.frame_skip_counter < self.depth_compute_interval and self.last_depth_map is not None:
            return self.last_depth_map
        
        self.frame_skip_counter = 0
        
        h, w = frame_bgr.shape[:2]
        target_size = (320, 240)
        frame_small = cv2.resize(frame_bgr, target_size)
        
        rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
        inp = transform(rgb).to(device)
        
        with torch.no_grad():
            pred = midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False
            ).squeeze()
        
        depth_map = pred.cpu().numpy()
        self.last_depth_map = depth_map
        return depth_map

    def update(self, gray, depth_map):
        img_height, img_width = gray.shape
        self.frame_count += 1
        
        self.lookahead_pixel = self._compute_lookahead_pixel(img_height, img_width)
        u_look, v_look = self.lookahead_pixel
        midas_depth = depth_map[v_look, u_look]
        self.lookahead_depth_curr = self._midas_to_metric_depth(midas_depth)
        
        if self.lookahead_depth_curr is None:
            self.prev_gray = gray
            self.prev_depth = depth_map
            return
        
        self.lookahead_world_pos = self._backproject_to_world(
            u_look, v_look, self.lookahead_depth_curr, self.pose
        )
        
        if self.prev_gray is not None:
            has_forward, has_rotation = self._detect_motion_state(gray)
            
            if has_rotation:
                self.motion_mode = "ROTATING"
                self.rest_frames_count = 0
            elif has_forward:
                self.motion_mode = "FORWARD"
                self.rest_frames_count = 0
            else:
                self.rest_frames_count += 1
                if self.rest_frames_count >= self.rest_frames_threshold:
                    self.motion_mode = "REST"
            
            if self.motion_mode == "ROTATING":
                prev_pt = np.array([[[u_look, v_look]]], dtype=np.float32)
                next_pt, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, prev_pt, None,
                    winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
                )
                
                if status is not None and status[0][0] == 1:
                    u_next, _ = next_pt[0][0]
                    delta_u = u_next - u_look
                    yaw_raw = delta_u / max(1e-6, self.fx)
                    yaw_delta = yaw_raw * self.ROTATION_SENSITIVITY
                    self.pose[2] += yaw_delta
                    self.rotation_magnitude = abs(math.degrees(yaw_delta))
                    self.translation_magnitude = 0.0
            
            elif self.motion_mode == "FORWARD":
                trans_step = self.FIXED_FORWARD_DISTANCE
                yaw = self.pose[2]
                self.pose[0] += trans_step * math.cos(yaw)
                self.pose[1] += trans_step * math.sin(yaw)
                
                old_dist = self.path_distance
                self.path_distance += trans_step
                
                if int(self.path_distance) > int(old_dist):
                    self.meter_markers.append({
                        'x': self.pose[0],
                        'y': self.pose[1],
                        'distance': int(self.path_distance)
                    })
                
                self.translation_magnitude = trans_step
                self.rotation_magnitude = 0.0
            
            else:
                self.translation_magnitude = 0.0
                self.rotation_magnitude = 0.0
        
        self.trajectory.append(self.pose.copy())
        self._detect_and_build_map(gray, depth_map, self.pose)
        
        if self.prev_keypoints is not None and len(self.prev_keypoints) > 0:
            corners = [(kp.pt[0], kp.pt[1]) for kp in self.prev_keypoints[:30]]
            self.prev_tracked_features = corners
        else:
            self.prev_tracked_features = None
        
        self.prev_gray = gray
        self.prev_depth = depth_map

    def draw_camera_view(self, frame_bgr, depth_map):
        """Fast rendering - show only recent features"""
        vis = frame_bgr.copy()
        
        # Only draw recent features
        for pt in self.recent_features:
            u, v = int(pt['u']), int(pt['v'])
            age = self.frame_count - pt['frame']
            alpha = 1.0 - (age / self.feature_lifetime_frames)
            brightness = int(255 * alpha)
            cv2.circle(vis, (u, v), 2, (brightness, brightness, 255), -1)
        
        if self.lookahead_pixel is not None:
            u_look, v_look = self.lookahead_pixel
            cv2.circle(vis, (u_look, v_look), 8, (0, 255, 255), -1)
            
            if self.lookahead_depth_curr is not None:
                cv2.putText(vis, f"{self.lookahead_depth_curr:.2f}m",
                           (u_look + 15, v_look - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        y = 20
        cv2.putText(vis, f"Feat: {self.feature_counts['total']}/{self.max_features_per_frame}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 20
        cv2.putText(vis, f"Show: {self.feature_counts['displayed']} | Map: {self.feature_counts['stored']}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 20
        
        state_colors = {"ROTATING": (0, 165, 255), "FORWARD": (0, 255, 0), "REST": (128, 128, 128)}
        state_color = state_colors.get(self.motion_mode, (255, 255, 255))
        
        if self.motion_mode == "ROTATING":
            state_text = f"ROT {self.rotation_magnitude:.1f}deg"
        elif self.motion_mode == "FORWARD":
            state_text = f"FWD {self.translation_magnitude:.3f}m"
        else:
            state_text = "REST"
            
        cv2.putText(vis, state_text, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
        
        return vis

    def draw_top_view(self, scale=100, size=900, fps=None):
        """Fast rendering - show recent features only during mapping"""
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        center_x, center_y = size // 2, size // 2
        
        cam_x, cam_y, cam_yaw = self.pose
        cam_screen_x = center_x
        cam_screen_y = int(size * 0.70)
        
        def project(x, y):
            dx = x - cam_x
            dy = y - cam_y
            sx = int(cam_screen_x + dx * scale)
            sy = int(cam_screen_y - dy * scale)
            return sx, sy
        
        # Draw only recent features (not all map!)
        for pt in self.recent_features:
            sx, sy = project(pt['x'], pt['y'])
            if 0 <= sx < size and 0 <= sy < size:
                brightness = int(np.clip(200.0 / (pt['z'] + 0.5), 40, 255))
                cv2.circle(canvas, (sx, sy), 2, (brightness, brightness, brightness), -1)
        
        # ORANGE trajectory
        if len(self.trajectory) > 1:
            points = []
            for (x, y, _) in self.trajectory:
                px, py = project(x, y)
                if 0 <= px < size and 0 <= py < size:
                    points.append((px, py))
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(canvas, points[i], points[i+1], (0, 165, 255), 3, cv2.LINE_AA)
        
        # Meter markers
        for marker in self.meter_markers:
            mx, my = project(marker['x'], marker['y'])
            if 0 <= mx < size and 0 <= my < size:
                cv2.circle(canvas, (mx, my), 4, (0, 255, 0), -1)
                cv2.putText(canvas, f"{marker['distance']}m", (mx + 8, my + 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Camera
        arrow_len = 35
        arrow_ex = int(cam_screen_x + arrow_len * math.cos(cam_yaw))
        arrow_ey = int(cam_screen_y - arrow_len * math.sin(cam_yaw))
        cv2.circle(canvas, (cam_screen_x, cam_screen_y), 6, (0, 255, 0), -1)
        
        arrow_colors = {"ROTATING": (0, 165, 255), "FORWARD": (0, 255, 0), "REST": (128, 128, 128)}
        arrow_color = arrow_colors.get(self.motion_mode, (255, 255, 255))
        cv2.arrowedLine(canvas, (cam_screen_x, cam_screen_y), (arrow_ex, arrow_ey),
                       arrow_color, 2, tipLength=0.3)
        
        y = 25
        if fps is not None:
            cv2.putText(canvas, f"FPS: {fps:.1f}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            y += 25
        
        cv2.putText(canvas, f"Dist: {self.path_distance:.2f}m", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 25
        
        cv2.putText(canvas, f"Map: {len(self.spatial_grid)} | Show: {len(self.recent_features)}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        return canvas

    def save_map_data(self, map_filename="map.pkl"):
        try:
            all_features = list(self.spatial_grid.values())
            
            print(f"💾 Saving {len(all_features)} features WITH distance info...")
            data = {
                'features': all_features,
                'trajectory': self.trajectory,
                'pose': self.pose,
                'distance': self.path_distance
            }
            with open(map_filename, "wb") as f:
                pickle.dump(data, f)
            self.last_map_pkl = map_filename
            print(f"✅ Saved!")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def run_slam(self, source=0, target_w=640, target_h=480, 
                 use_rover_stream=False, rover_url=None):
        if use_rover_stream and rover_url:
            cap = RoverStreamCapture(rover_url)
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            return
        
        print("\n" + "="*70)
        print("🚀 FAST SLAM - Dynamic Display + Accurate Pose")
        print("="*70)
        print(f"✅ Features: {self.min_features_per_frame}-{self.max_features_per_frame}/frame")
        print(f"✅ Display: {self.max_display_features} recent (fast!)")
        print(f"✅ Store: {self.max_map_features} total (accurate!)")
        print(f"✅ Depth skip: Every {self.depth_compute_interval} frames")
        print("\n⌨️ [ESC] Quit | [S] Save & test")
        print("="*70 + "\n")
        
        fps = 0.0
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (target_w, target_h))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth = self.depth_map(frame)
            
            self.update(gray, depth)
            
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt) if fps > 0 else 1.0 / dt
            
            vis = self.draw_camera_view(frame, depth)
            cv2.imshow("Camera View", vis)
            
            top_map = self.draw_top_view(scale=100, size=900, fps=fps)
            cv2.imshow("Top View Map", top_map)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('s') or key == ord('S'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                map_img = self.draw_top_view(scale=100, size=900, fps=None)
                map_filename = f"slam_map_{timestamp}.png"
                cv2.imwrite(map_filename, map_img)
                self.last_map_filename = map_filename
                self.save_map_data(f"map_{timestamp}.pkl")
                
                print("\n🧪 Test pose? [Y/N]")
                
                while True:
                    choice_key = cv2.waitKey(0) & 0xFF
                    if choice_key == ord('y') or choice_key == ord('Y'):
                        cv2.destroyAllWindows()
                        cap.release()
                        
                        print("\n📸 Image:")
                        test_img_path = input().strip()
                        
                        if os.path.exists(test_img_path):
                            estimate_pose_accurate(test_img_path, self.last_map_pkl)
                        else:
                            print("❌ Not found")
                        
                        return
                    elif choice_key == ord('n') or choice_key == ord('N'):
                        break
                    elif choice_key == 27:
                        cap.release()
                        cv2.destroyAllWindows()
                        return
        
        cap.release()
        cv2.destroyAllWindows()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_map = self.draw_top_view(scale=100, size=900, fps=None)
        cv2.imwrite(f"slam_map_{timestamp}.png", final_map)
        self.save_map_data(f"map_{timestamp}.pkl")


# ========== ACCURATE POSE ESTIMATION - Multi-candidate voting ==========

def estimate_pose_accurate(img_path, map_data_path):
    """ACCURATE: Vote for best path segment using distance-filtered features"""
    print("\n" + "="*70)
    print("🎯 ACCURATE POSE ESTIMATION")
    print("="*70)
    
    # Load map
    try:
        with open(map_data_path, "rb") as f:
            map_data = pickle.load(f)
        
        all_features = map_data.get('features', [])
        trajectory = map_data.get('trajectory', [])
        
        print(f"✅ {len(all_features)} features loaded")
        print(f"✅ {len(trajectory)} trajectory points")
        
        has_distance = 'cam_distance' in all_features[0] if all_features else False
        if not has_distance:
            print("❌ Map missing distance info! Rebuild map.")
            return
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Load and process image
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Cannot load image")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get depth
    print("🔧 Computing depth...")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_small = cv2.resize(rgb, (320, 240))
    inp = transform(rgb_small).to(device)
    
    with torch.no_grad():
        pred = midas(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img.shape[:2],
            mode="bilinear",
            align_corners=False
        ).squeeze()
    
    test_depth_map = pred.cpu().numpy()
    
    def midas_to_metric(val):
        return DEPTH_SCALE / (val + 1e-6) if val >= 1e-3 else None
    
    # Preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=5)
    
    # Extract features
    orb = cv2.ORB_create(nfeatures=300, fastThreshold=10)
    kp, desc = orb.detectAndCompute(enhanced, None)
    
    if desc is None or len(desc) == 0:
        print("❌ No features in test image")
        return
    
    print(f"✅ {len(kp)} test features extracted")
    
    # Get depth for each keypoint
    kp_depths = []
    for k in kp:
        u, v = int(k.pt[0]), int(k.pt[1])
        if 0 <= v < test_depth_map.shape[0] and 0 <= u < test_depth_map.shape[1]:
            midas_d = test_depth_map[v, u]
            metric_d = midas_to_metric(midas_d)
            kp_depths.append(metric_d if metric_d is not None else 0.0)
        else:
            kp_depths.append(0.0)
    
    # Match features
    map_descs = np.array([f['descriptor'] for f in all_features])
    print(f"✅ Matching against {len(map_descs)} map features...")
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc, map_descs, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    print(f"✅ {len(good_matches)} descriptor matches")
    
    # Distance filtering (STRICT ±30cm)
    distance_tolerance = 0.30
    filtered_matches = []
    
    for m in good_matches:
        map_feat = all_features[m.trainIdx]
        map_distance = map_feat.get('cam_distance', 0.0)
        test_distance = kp_depths[m.queryIdx]
        
        if test_distance > 0.1:
            dist_diff = abs(map_distance - test_distance)
            if dist_diff < distance_tolerance:
                filtered_matches.append(m)
    
    print(f"✅ {len(filtered_matches)} distance-filtered matches")
    
    if len(filtered_matches) < 8:
        print(f"❌ Insufficient matches: {len(filtered_matches)}/8")
        return
    
    # Sort and keep best
    filtered_matches = sorted(filtered_matches, key=lambda m: m.distance)
    best_matches = filtered_matches[:min(50, len(filtered_matches))]
    
    # Vote for path segments
    print("🗳️  Voting for path segment...")
    
    matched_features = [all_features[m.trainIdx] for m in best_matches]
    
    # Group features by their world position
    feature_positions = [(f['x'], f['y']) for f in matched_features]
    
    # Find cluster center
    xs = [f[0] for f in feature_positions]
    ys = [f[1] for f in feature_positions]
    cluster_x = np.median(xs)
    cluster_y = np.median(ys)
    
    print(f"   Feature cluster: ({cluster_x:.3f}, {cluster_y:.3f})")
    
    # Find nearest path segment
    if len(trajectory) < 2:
        print("❌ Insufficient trajectory")
        return
    
    min_dist = float('inf')
    best_seg_idx = 0
    best_pos = (cluster_x, cluster_y)
    
    for i in range(len(trajectory) - 1):
        p1 = np.array([trajectory[i][0], trajectory[i][1]])
        p2 = np.array([trajectory[i+1][0], trajectory[i+1][1]])
        
        # Project cluster center onto segment
        v = p2 - p1
        w = np.array([cluster_x, cluster_y]) - p1
        
        if np.dot(v, v) < 1e-8:
            proj = p1
        else:
            t = np.dot(w, v) / np.dot(v, v)
            t = np.clip(t, 0, 1)
            proj = p1 + t * v
        
        dist = np.linalg.norm(np.array([cluster_x, cluster_y]) - proj)
        
        if dist < min_dist:
            min_dist = dist
            best_seg_idx = i
            best_pos = (proj[0], proj[1])
    
    x_pred, y_pred = best_pos
    
    # Estimate yaw
    if best_seg_idx < len(trajectory) - 1:
        p1 = trajectory[best_seg_idx]
        p2 = trajectory[best_seg_idx + 1]
        yaw_pred = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    else:
        yaw_pred = trajectory[-1][2] if len(trajectory[-1]) > 2 else 0.0
    
    # Confidence
    avg_distance = np.mean([m.distance for m in best_matches])
    confidence = max(0.0, 1.0 - (avg_distance / 128.0))
    
    print(f"\n✅ FINAL POSE:")
    print(f"   Position: ({x_pred:.3f}, {y_pred:.3f})")
    print(f"   Path segment: {best_seg_idx}/{len(trajectory)-1}")
    print(f"   Projection distance: {min_dist:.3f}m")
    print(f"   Yaw: {math.degrees(yaw_pred):.1f}°")
    print(f"   Confidence: {confidence*100:.1f}%")
    print(f"   Matches used: {len(best_matches)}")
    
    # Visualize with ALL map features
    visualize_accurate(x_pred, y_pred, yaw_pred, confidence, 
                      all_features, img, kp, best_matches, 
                      trajectory, best_seg_idx, cluster_x, cluster_y)


def visualize_accurate(x, y, yaw, confidence, all_features, test_image, 
                       kp, matches, trajectory, seg_idx, cluster_x, cluster_y):
    """Visualize with FULL map (not just recent)"""
    size = 900
    scale = 100
    
    map_canvas = np.zeros((size, size, 3), dtype=np.uint8)
    center_x, center_y = size // 2, size // 2
    cam_screen_x = center_x
    cam_screen_y = int(size * 0.70)
    
    def project(px, py):
        dx = px - x
        dy = py - y
        sx = int(cam_screen_x + dx * scale)
        sy = int(cam_screen_y - dy * scale)
        return sx, sy
    
    # Draw ALL map features (full map)
    for i, feat in enumerate(all_features):
        if i % 2 == 0:  # Sample every other for speed
            sx, sy = project(feat['x'], feat['y'])
            if 0 <= sx < size and 0 <= sy < size:
                brightness = int(np.clip(200.0 / (feat['z'] + 0.5), 40, 255))
                cv2.circle(map_canvas, (sx, sy), 1, (brightness, brightness, brightness), -1)
    
    # ORANGE trajectory
    if len(trajectory) > 1:
        traj_points = []
        for pose in trajectory:
            if len(pose) >= 2:
                tx, ty = pose[0], pose[1]
                px, py = project(tx, ty)
                if 0 <= px < size and 0 <= py < size:
                    traj_points.append((px, py))
        
        if len(traj_points) > 1:
            for i in range(len(traj_points) - 1):
                cv2.line(map_canvas, traj_points[i], traj_points[i+1], 
                        (0, 165, 255), 5, cv2.LINE_AA)
    
    # Matched features (green)
    for m in matches:
        feat = all_features[m.trainIdx]
        sx, sy = project(feat['x'], feat['y'])
        if 0 <= sx < size and 0 <= sy < size:
            cv2.circle(map_canvas, (sx, sy), 4, (0, 255, 0), -1)
    
    # Feature cluster center (yellow)
    cluster_sx, cluster_sy = project(cluster_x, cluster_y)
    if 0 <= cluster_sx < size and 0 <= cluster_sy < size:
        cv2.circle(map_canvas, (cluster_sx, cluster_sy), 12, (0, 255, 255), 2)
        cv2.circle(map_canvas, (cluster_sx, cluster_sy), 3, (0, 255, 255), -1)
    
    # PINK pose on path
    PINK = (203, 192, 255)
    arrow_len = 50
    arrow_ex = int(cam_screen_x + arrow_len * math.cos(yaw))
    arrow_ey = int(cam_screen_y - arrow_len * math.sin(yaw))
    
    cv2.circle(map_canvas, (cam_screen_x, cam_screen_y), 18, PINK, -1)
    cv2.circle(map_canvas, (cam_screen_x, cam_screen_y), 24, PINK, 4)
    cv2.arrowedLine(map_canvas, (cam_screen_x, cam_screen_y), (arrow_ex, arrow_ey),
                   PINK, 5, tipLength=0.3)
    
    # Info
    cv2.putText(map_canvas, "ACCURATE POSE", (10, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, PINK, 3)
    cv2.putText(map_canvas, f"Pos: ({x:.2f}, {y:.2f})", (10, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(map_canvas, f"Segment: {seg_idx}/{len(trajectory)-1}", (10, 105),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(map_canvas, f"Yaw: {math.degrees(yaw):.1f}deg", (10, 135),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(map_canvas, f"Conf: {confidence*100:.1f}%", (10, 165),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(map_canvas, f"Matches: {len(matches)}", (10, 195),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Legend
    y_legend = size - 120
    cv2.putText(map_canvas, "ORANGE = Path", (10, y_legend),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(map_canvas, "PINK = You (on path)", (10, y_legend + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, PINK, 2)
    cv2.putText(map_canvas, "GREEN = Matches", (10, y_legend + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(map_canvas, "YELLOW = Feature cluster", (10, y_legend + 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Test image
    test_vis = test_image.copy()
    for m in matches:
        pt = kp[m.queryIdx].pt
        cv2.circle(test_vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
    
    test_vis = cv2.resize(test_vis, (640, 480))
    cv2.putText(test_vis, "TEST IMAGE", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, PINK, 2)
    
    cv2.imshow("Test Image", test_vis)
    cv2.imshow("Map - Accurate Pose", map_canvas)
    
    print("\n" + "="*70)
    print("📍 SUCCESS - Accurate Position!")
    print("="*70)
    print("🟠 ORANGE = Traveled path")
    print("💗 PINK = Your position (on path)")
    print("🟢 GREEN = Distance-filtered matches")
    print("🟡 YELLOW = Feature cluster center")
    print("\n⌨️  Press any key...")
    print("="*70)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"pose_accurate_{timestamp}.png", map_canvas)
    print(f"\n✅ Saved: pose_accurate_{timestamp}.png")


# ========== MENU ==========

def main_menu():
    print("\n" + "="*70)
    print("🚀 FAST SLAM - Dynamic Display + Accurate Pose")
    print("="*70)
    print("1. Build Map (fast rendering)")
    print("2. Test Pose (accurate, full map)")
    print("="*70)
    
    choice = input("Choose [1/2]: ").strip()
    
    if choice == '1':
        build_map_menu()
    elif choice == '2':
        estimate_menu()


def build_map_menu():
    print("\n📹 BUILD MAP")
    print("0 - Rover")
    print("1 - Webcam")
    print("2 - Video")
    
    choice = input("Choose: ").strip()
    
    rot_sens = 1.1
    rotation_sensitivity = float(rot_sens) if rot_sens else 3.0
    
    depth_skip = input("Depth skip [2]: ").strip()
    depth_compute_interval = int(depth_skip) if depth_skip else 2
    
    slam = FastSLAM(
        fx=600.0, fy=600.0, cx=320.0, cy=240.0,
        min_features_per_frame=50,
        max_features_per_frame=120,
        rotation_sensitivity=rotation_sensitivity,
        feature_lifetime_frames=30,
        max_depth_threshold=3.0,
        spatial_grid_size=0.10,
        depth_compute_interval=depth_compute_interval,
        max_display_features=200  # Show only 200 recent
    )
    
    if choice == '0':
        slam.run_slam(source=None, use_rover_stream=True, 
                     rover_url="http://10.47.11.127:8080/video_feed")
    elif choice == '1':
        slam.run_slam(source=0)
    elif choice == '2':
        path = "loops.mp4"
        if os.path.exists(path):
            slam.run_slam(source=path)


def estimate_menu():
    print("\n🔍 TEST POSE (Accurate)")
    
    import glob
    maps = sorted(glob.glob("map_*.pkl"), reverse=True)
    
    if not maps:
        print("❌ No maps!")
        return
    
    print(f"✅ Using: {maps[0]}")
    
    img = input("📸 Image: ").strip()
    
    if os.path.exists(img):
        estimate_pose_accurate(img, maps[0])
    else:
        print("❌ Not found")


if __name__ == "__main__":
    main_menu()