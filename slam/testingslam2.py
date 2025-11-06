import cv2
import torch
import math
import numpy as np
import time
import os
import pickle
import requests
from collections import defaultdict
from scipy.spatial import cKDTree

# =============== Load Scaled Depth Factor ===============
try:
    DEPTH_SCALE = np.load("depth_scale_factor.npy").item()
except Exception:
    DEPTH_SCALE = 5.0

# =============== MiDaS Depth Model on GPU ===============
print("🔧 Initializing MiDaS...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

if torch.cuda.is_available():
    print(f"✅ CUDA - GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  CPU MODE")


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


class EnhancedSLAM:
    """SLAM with keyframe detection and matching"""
    
    def __init__(self, fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                 lookahead_height_ratio=0.9,
                 min_features_per_frame=50, max_features_per_frame=150,
                 rotation_sensitivity=3.0, feature_lifetime_frames=50,
                 max_depth_threshold=3.0, spatial_grid_size=0.10,
                 keyframe_translation_threshold=0.15, keyframe_rotation_threshold=15.0):
        
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        
        # Keyframe parameters
        self.keyframe_translation_threshold = keyframe_translation_threshold  # meters
        self.keyframe_rotation_threshold = math.radians(keyframe_rotation_threshold)  # radians
        self.keyframes = []  # List of keyframe dictionaries
        self.last_keyframe_pose = None
        self.current_keyframe_id = 0
        self.keyframe_detected_this_frame = False
        
        self.pose = np.zeros(3, dtype=float)
        self.trajectory = []
        self.path_distance = 0.0
        self.meter_markers = []
        
        self.motion_mode = "REST"
        self.FIXED_FORWARD_DISTANCE = 0.004
        self.ROTATION_SENSITIVITY = rotation_sensitivity
        self.rest_frames_count = 0
        self.rest_frames_threshold = 10
        
        # STRICT feature limits
        self.min_features_per_frame = min_features_per_frame
        self.max_features_per_frame = max_features_per_frame
        self.current_feature_target = max_features_per_frame
        
        # Spatial grid for UNIQUE features only (10cm grid)
        self.spatial_grid_size = spatial_grid_size
        self.spatial_grid = {}  # Only ONE feature per grid cell
        self.max_map_features = 15000  # Hard limit on total features
        
        self.feature_lifetime_frames = feature_lifetime_frames
        self.visible_map_points = []
        
        self.max_depth_threshold = max_depth_threshold
        
        # Fixed loop closure (less sensitive)
        self.loop_closure_history = []
        self.loop_closure_threshold = 0.90  # Higher = less sensitive
        self.loop_closure_cooldown = 0
        self.loop_closure_cooldown_frames = 100  # Don't trigger for 100 frames after detection
        
        # Minimal preprocessing
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # ORB with good quality
        self.orb = cv2.ORB_create(
            nfeatures=500,  # Detect 500, select best 50-150
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=10,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=10
        )
        
        # Matchers
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, 
                          key_size=20, multi_probe_level=2)
        search_params = dict(checks=100)
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
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
        self.visible_features = []
        self.feature_counts = {'total': 0, 'valid_depth': 0, 'displayed': 0, 'stored': 0}
        self.rotation_magnitude = 0.0
        self.translation_magnitude = 0.0
        
        self.last_map_filename = None
        self.last_map_pkl = None
        
        self.loop_closure_detected = False

    def _preprocess_frame(self, gray):
        """Minimal preprocessing"""
        enhanced = self.clahe.apply(gray)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=5)
        return enhanced

    def _spatial_grid_key(self, x, y):
        """Generate spatial grid key"""
        return (
            int(x / self.spatial_grid_size),
            int(y / self.spatial_grid_size)
        )

    def _add_to_spatial_grid(self, x, y, z, descriptor, quality, frame):
        """Add feature to spatial grid (only one per cell)"""
        key = self._spatial_grid_key(x, y)
        
        # Check if cell already occupied
        if key in self.spatial_grid:
            # Keep higher quality feature
            existing = self.spatial_grid[key]
            if quality > existing['quality']:
                self.spatial_grid[key] = {
                    'x': x, 'y': y, 'z': z,
                    'descriptor': descriptor,
                    'quality': quality,
                    'frame': frame
                }
        else:
            # New cell
            if len(self.spatial_grid) < self.max_map_features:
                self.spatial_grid[key] = {
                    'x': x, 'y': y, 'z': z,
                    'descriptor': descriptor,
                    'quality': quality,
                    'frame': frame
                }

    def _should_create_keyframe(self):
        """Determine if current frame should be a keyframe"""
        if self.last_keyframe_pose is None:
            return True
        
        # Calculate translation distance
        dx = self.pose[0] - self.last_keyframe_pose[0]
        dy = self.pose[1] - self.last_keyframe_pose[1]
        translation = math.sqrt(dx*dx + dy*dy)
        
        # Calculate rotation change
        rotation = abs(self.pose[2] - self.last_keyframe_pose[2])
        
        return (translation >= self.keyframe_translation_threshold or 
                rotation >= self.keyframe_rotation_threshold)

    def _create_keyframe(self, gray, descriptors, keypoints):
        """Create a new keyframe"""
        if descriptors is None or len(descriptors) == 0:
            return
        
        keyframe = {
            'id': self.current_keyframe_id,
            'pose': self.pose.copy(),
            'frame_number': self.frame_count,
            'descriptors': descriptors.copy(),
            'keypoints': keypoints,
            'image': gray.copy()  # Store for visualization (optional, can be omitted to save memory)
        }
        
        self.keyframes.append(keyframe)
        self.last_keyframe_pose = self.pose.copy()
        self.current_keyframe_id += 1
        self.keyframe_detected_this_frame = True
        
        print(f"🔑 Keyframe #{self.current_keyframe_id} @ ({self.pose[0]:.2f}, {self.pose[1]:.2f})")

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
        """Fixed loop closure - less sensitive, with cooldown"""
        # Cooldown check
        if self.loop_closure_cooldown > 0:
            self.loop_closure_cooldown -= 1
            return False
        
        if current_pattern is None or len(self.loop_closure_history) < 5:
            return False
        
        min_frame_gap = 100  # Increased from 50
        current_frame = self.frame_count
        
        for past_frame, past_pattern in self.loop_closure_history[-20:]:  # Only check recent 20
            if current_frame - past_frame < min_frame_gap:
                continue
            
            if len(current_pattern) == len(past_pattern):
                similarity = np.corrcoef(current_pattern, past_pattern)[0, 1]
                if similarity > self.loop_closure_threshold:
                    self.loop_closure_cooldown = self.loop_closure_cooldown_frames
                    return True
        
        return False

    def _adjust_feature_target(self):
        """Keep target within 50-150 range"""
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

    def _update_visible_features(self):
        self.visible_map_points = [
            pt for pt in self.visible_map_points 
            if self.frame_count - pt['frame'] < self.feature_lifetime_frames
        ]

    def _detect_and_build_map(self, gray, depth_map, pose):
        """STRICT 50-150 feature limit with spatial uniqueness + Keyframe creation"""
        self.keyframe_detected_this_frame = False
        
        enhanced = self._preprocess_frame(gray)
        img_height, img_width = gray.shape
        
        # Focus on bottom 2/3 where ground features are
        mask = np.zeros_like(enhanced, dtype=np.uint8)
        h_start = int(img_height * 0.33)
        mask[h_start:, :] = 255
        
        # Detect features
        kp_orb, desc_orb = self.orb.detectAndCompute(enhanced, mask=mask)
        
        if kp_orb is None or desc_orb is None or len(kp_orb) == 0:
            self.feature_counts['total'] = 0
            self.feature_counts['valid_depth'] = 0
            return []
        
        # Sort by quality and limit to max_features_per_frame
        kp_desc_pairs = list(zip(kp_orb, desc_orb))
        kp_desc_pairs.sort(key=lambda x: -x[0].response)
        
        # ENFORCE STRICT LIMIT
        n_features = min(len(kp_desc_pairs), self.max_features_per_frame)
        kp_desc_pairs = kp_desc_pairs[:n_features]
        
        kp_orb = [kp for kp, _ in kp_desc_pairs]
        desc_orb = np.array([desc for _, desc in kp_desc_pairs])
        
        # Check if should create keyframe
        if self._should_create_keyframe():
            self._create_keyframe(gray, desc_orb, kp_orb)
        
        # Loop closure check
        depth_pattern = self._extract_depth_pattern(depth_map, kp_orb)
        
        if self._check_loop_closure(depth_pattern):
            if not self.loop_closure_detected:
                print("🔄 LOOP CLOSURE")
                self.loop_closure_detected = True
        else:
            self.loop_closure_detected = False
        
        if depth_pattern is not None:
            self.loop_closure_history.append((self.frame_count, depth_pattern))
            if len(self.loop_closure_history) > 50:  # Keep only last 50
                self.loop_closure_history.pop(0)
        
        valid_depth_count = 0
        new_features = []
        all_features = []
        
        # Process features - add to spatial grid
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
                    
                    # Add to spatial grid (automatically handles duplicates)
                    self._add_to_spatial_grid(
                        x_world, y_world, metric_depth, 
                        descriptor, quality, self.frame_count
                    )
                    
                    # Add to visible (for display)
                    self.visible_map_points.append({
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
        
        self._update_visible_features()
        self.feature_counts['displayed'] = len(self.visible_map_points)
        
        self.current_features = all_features
        self.visible_features = [(int(pt['u']), int(pt['v'])) for pt in self.visible_map_points]
        
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
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
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
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = transform(rgb).to(device)
        
        with torch.no_grad():
            pred = midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()
        
        return pred.cpu().numpy()

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
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
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
            corners = [(kp.pt[0], kp.pt[1]) for kp in self.prev_keypoints[:50]]
            self.prev_tracked_features = corners
        else:
            self.prev_tracked_features = None
        
        self.prev_gray = gray
        self.prev_depth = depth_map

    def draw_camera_view(self, frame_bgr, depth_map):
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
        vis = cv2.addWeighted(frame_bgr, 0.4, depth_colored, 0.6, 0)
        
        for pt in self.visible_map_points:
            u, v = int(pt['u']), int(pt['v'])
            age = self.frame_count - pt['frame']
            alpha = 1.0 - (age / self.feature_lifetime_frames)
            brightness = int(255 * alpha)
            cv2.circle(vis, (u, v), 3, (brightness, brightness, 255), -1)
        
        if self.lookahead_pixel is not None:
            u_look, v_look = self.lookahead_pixel
            cv2.circle(vis, (u_look, v_look), 10, (0, 255, 255), -1)
            
            if self.lookahead_depth_curr is not None:
                cv2.putText(vis, f"{self.lookahead_depth_curr:.2f}m",
                           (u_look + 20, v_look - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show keyframe indicator
        if self.keyframe_detected_this_frame:
            cv2.putText(vis, "🔑 KEYFRAME", (vis.shape[1] - 180, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        y = 25
        cv2.putText(vis, f"Features: {self.feature_counts['total']}/{self.max_features_per_frame}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25
        cv2.putText(vis, f"Map: {self.feature_counts['stored']}/{self.max_map_features}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25
        cv2.putText(vis, f"Keyframes: {len(self.keyframes)}",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        y += 25
        
        state_colors = {"ROTATING": (0, 165, 255), "FORWARD": (0, 255, 0), "REST": (128, 128, 128)}
        state_color = state_colors.get(self.motion_mode, (255, 255, 255))
        
        if self.motion_mode == "ROTATING":
            state_text = f"ROTATING ({self.rotation_magnitude:.2f}deg)"
        elif self.motion_mode == "FORWARD":
            state_text = f"FORWARD ({self.translation_magnitude:.4f}m)"
        else:
            state_text = "REST"
            
        cv2.putText(vis, state_text, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        return vis

    def draw_top_view(self, scale=100, size=900, fps=None):
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
        
        # Draw map features
        for feat in self.spatial_grid.values():
            sx, sy = project(feat['x'], feat['y'])
            if 0 <= sx < size and 0 <= sy < size:
                brightness = int(np.clip(200.0 / (feat['z'] + 0.5), 40, 255))
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
        
        # Draw keyframe positions with BLUE perpendicular lines
        for kf in self.keyframes:
            kf_x, kf_y, kf_yaw = kf['pose']
            kf_sx, kf_sy = project(kf_x, kf_y)
            
            if 0 <= kf_sx < size and 0 <= kf_sy < size:
                # Draw perpendicular line (90 degrees to yaw direction)
                line_length = 30
                perp_yaw = kf_yaw + math.pi / 2
                
                # Line endpoints
                line_x1 = int(kf_sx + line_length * math.cos(perp_yaw))
                line_y1 = int(kf_sy - line_length * math.sin(perp_yaw))
                line_x2 = int(kf_sx - line_length * math.cos(perp_yaw))
                line_y2 = int(kf_sy + line_length * math.sin(perp_yaw))
                
                # Draw blue perpendicular line
                cv2.line(canvas, (line_x1, line_y1), (line_x2, line_y2), (255, 0, 0), 3, cv2.LINE_AA)
                
                # Draw small circle at keyframe center
                cv2.circle(canvas, (kf_sx, kf_sy), 5, (255, 0, 0), -1)
        
        # Meter markers
        for marker in self.meter_markers:
            mx, my = project(marker['x'], marker['y'])
            if 0 <= mx < size and 0 <= my < size:
                cv2.circle(canvas, (mx, my), 6, (0, 255, 0), -1)
                cv2.putText(canvas, f"{marker['distance']}m", (mx + 12, my + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Camera with keyframe indicator
        arrow_len = 40
        arrow_ex = int(cam_screen_x + arrow_len * math.cos(cam_yaw))
        arrow_ey = int(cam_screen_y - arrow_len * math.sin(cam_yaw))
        
        # Draw current keyframe blue line if just detected
        if self.keyframe_detected_this_frame:
            line_length = 30
            perp_yaw = cam_yaw + math.pi / 2
            line_x1 = int(cam_screen_x + line_length * math.cos(perp_yaw))
            line_y1 = int(cam_screen_y - line_length * math.sin(perp_yaw))
            line_x2 = int(cam_screen_x - line_length * math.cos(perp_yaw))
            line_y2 = int(cam_screen_y + line_length * math.sin(perp_yaw))
            cv2.line(canvas, (line_x1, line_y1), (line_x2, line_y2), (255, 0, 0), 4, cv2.LINE_AA)
        
        cv2.circle(canvas, (cam_screen_x, cam_screen_y), 8, (0, 255, 0), -1)
        
        arrow_colors = {"ROTATING": (0, 165, 255), "FORWARD": (0, 255, 0), "REST": (128, 128, 128)}
        arrow_color = arrow_colors.get(self.motion_mode, (255, 255, 255))
        cv2.arrowedLine(canvas, (cam_screen_x, cam_screen_y), (arrow_ex, arrow_ey),
                       arrow_color, 3, tipLength=0.3)
        
        y = 30
        if fps is not None:
            cv2.putText(canvas, f"FPS: {fps:.1f}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            y += 30
        
        cv2.putText(canvas, f"Distance: {self.path_distance:.2f}m", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 30
        
        cv2.putText(canvas, f"Points: {len(self.spatial_grid)}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y += 30
        
        cv2.putText(canvas, f"Keyframes: {len(self.keyframes)}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        return canvas

    def save_map_data(self, map_filename="map.pkl"):
        try:
            # Convert spatial grid to list
            all_features = list(self.spatial_grid.values())
            
            print(f"💾 Saving {len(all_features)} features + {len(self.keyframes)} keyframes...")
            data = {
                'features': all_features,
                'keyframes': self.keyframes,
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
        print("🎯 SLAM with Keyframes")
        print("="*70)
        print(f"✅ Feature limit: {self.min_features_per_frame}-{self.max_features_per_frame}/frame")
        print(f"✅ Keyframe thresholds: {self.keyframe_translation_threshold}m translation, {math.degrees(self.keyframe_rotation_threshold):.1f}° rotation")
        print(f"✅ Blue lines show keyframe positions (perpendicular to heading)")
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
                            estimate_pose_with_keyframes(test_img_path, self.last_map_pkl, self.trajectory)
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


# ========== KEYFRAME-BASED POSE ESTIMATION ==========

def estimate_pose_with_keyframes(img_path, map_data_path, trajectory_from_map=None):
    """Pose estimation using keyframe matching"""
    print("\n" + "="*70)
    print("🔍 KEYFRAME-BASED POSE ESTIMATION")
    print("="*70)
    
    # Load map
    try:
        with open(map_data_path, "rb") as f:
            map_data = pickle.load(f)
        
        keyframes = map_data.get('keyframes', [])
        all_features = map_data.get('features', [])
        trajectory = map_data.get('trajectory', trajectory_from_map or [])
        
        print(f"✅ {len(keyframes)} keyframes loaded")
        print(f"✅ {len(all_features)} features loaded")
        print(f"✅ {len(trajectory)} trajectory points")
        
        if len(keyframes) == 0:
            print("❌ No keyframes found! Falling back to feature matching...")
            estimate_pose_advanced(img_path, map_data_path, trajectory)
            return
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Load test image
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Cannot load image")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Same preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=5)
    
    # Extract features
    orb = cv2.ORB_create(nfeatures=500, fastThreshold=10)
    kp, desc = orb.detectAndCompute(enhanced, None)
    
    if desc is None or len(desc) == 0:
        print("❌ No features in test image")
        return
    
    print(f"✅ {len(kp)} features extracted from test image")
    
    # Match against each keyframe
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    keyframe_matches = []
    
    for kf in keyframes:
        kf_desc = kf['descriptors']
        
        if kf_desc is None or len(kf_desc) == 0:
            continue
        
        matches = bf.knnMatch(desc, kf_desc, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) >= 4:
            avg_distance = np.mean([m.distance for m in good_matches])
            keyframe_matches.append({
                'keyframe': kf,
                'matches': good_matches,
                'match_count': len(good_matches),
                'avg_distance': avg_distance,
                'score': len(good_matches) / (avg_distance + 1)  # Higher is better
            })
    
    if len(keyframe_matches) == 0:
        print("❌ No keyframe matches found")
        return
    
    # Sort by score (best match first)
    keyframe_matches.sort(key=lambda x: -x['score'])
    
    best_match = keyframe_matches[0]
    best_kf = best_match['keyframe']
    best_matches = best_match['matches']
    
    print(f"\n✅ Best keyframe match: #{best_kf['id']}")
    print(f"   Matches: {best_match['match_count']}")
    print(f"   Avg distance: {best_match['avg_distance']:.1f}")
    print(f"   Score: {best_match['score']:.2f}")
    
    # Use keyframe pose as estimate
    x_pred, y_pred, yaw_pred = best_kf['pose']
    
    confidence = min(1.0, best_match['match_count'] / 100.0) * (1.0 - best_match['avg_distance'] / 128.0)
    confidence = max(0.0, confidence)
    
    print(f"\n✅ ESTIMATED POSE (from keyframe #{best_kf['id']}):")
    print(f"   Position: ({x_pred:.3f}, {y_pred:.3f})")
    print(f"   Yaw: {math.degrees(yaw_pred):.1f}°")
    print(f"   Confidence: {confidence*100:.1f}%")
    
    # Visualize
    visualize_keyframe_match(x_pred, y_pred, yaw_pred, confidence,
                            keyframes, img, kp, best_matches, best_kf, trajectory)


def visualize_keyframe_match(x, y, yaw, confidence, keyframes, test_image,
                             kp, matches, matched_kf, trajectory):
    """Visualize keyframe-based localization"""
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
                        (0, 165, 255), 3, cv2.LINE_AA)
    
    # Draw all keyframes (gray)
    for kf in keyframes:
        kf_x, kf_y, kf_yaw = kf['pose']
        kf_sx, kf_sy = project(kf_x, kf_y)
        
        if 0 <= kf_sx < size and 0 <= kf_sy < size:
            line_length = 30
            perp_yaw = kf_yaw + math.pi / 2
            
            line_x1 = int(kf_sx + line_length * math.cos(perp_yaw))
            line_y1 = int(kf_sy - line_length * math.sin(perp_yaw))
            line_x2 = int(kf_sx - line_length * math.cos(perp_yaw))
            line_y2 = int(kf_sy + line_length * math.sin(perp_yaw))
            
            # Gray lines for non-matched keyframes
            cv2.line(map_canvas, (line_x1, line_y1), (line_x2, line_y2),
                    (100, 100, 100), 2, cv2.LINE_AA)
            cv2.circle(map_canvas, (kf_sx, kf_sy), 4, (100, 100, 100), -1)
    
    # Highlight matched keyframe in GREEN
    matched_x, matched_y, matched_yaw = matched_kf['pose']
    matched_sx, matched_sy = project(matched_x, matched_y)
    
    if 0 <= matched_sx < size and 0 <= matched_sy < size:
        line_length = 35
        perp_yaw = matched_yaw + math.pi / 2
        
        line_x1 = int(matched_sx + line_length * math.cos(perp_yaw))
        line_y1 = int(matched_sy - line_length * math.sin(perp_yaw))
        line_x2 = int(matched_sx - line_length * math.cos(perp_yaw))
        line_y2 = int(matched_sy + line_length * math.sin(perp_yaw))
        
        # GREEN line for matched keyframe
        cv2.line(map_canvas, (line_x1, line_y1), (line_x2, line_y2),
                (0, 255, 0), 5, cv2.LINE_AA)
        cv2.circle(map_canvas, (matched_sx, matched_sy), 8, (0, 255, 0), -1)
    
    # PINK pose (current estimated position)
    PINK = (203, 192, 255)
    arrow_len = 50
    arrow_ex = int(cam_screen_x + arrow_len * math.cos(yaw))
    arrow_ey = int(cam_screen_y - arrow_len * math.sin(yaw))
    
    # Draw perpendicular blue line at estimated pose
    line_length = 35
    perp_yaw = yaw + math.pi / 2
    line_x1 = int(cam_screen_x + line_length * math.cos(perp_yaw))
    line_y1 = int(cam_screen_y - line_length * math.sin(perp_yaw))
    line_x2 = int(cam_screen_x - line_length * math.cos(perp_yaw))
    line_y2 = int(cam_screen_y + line_length * math.sin(perp_yaw))
    cv2.line(map_canvas, (line_x1, line_y1), (line_x2, line_y2), (255, 0, 0), 5, cv2.LINE_AA)
    
    cv2.circle(map_canvas, (cam_screen_x, cam_screen_y), 18, PINK, -1)
    cv2.circle(map_canvas, (cam_screen_x, cam_screen_y), 24, PINK, 4)
    cv2.arrowedLine(map_canvas, (cam_screen_x, cam_screen_y), (arrow_ex, arrow_ey),
                   PINK, 5, tipLength=0.3)
    
    # Info
    cv2.putText(map_canvas, "KEYFRAME LOCALIZATION", (10, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, PINK, 3)
    cv2.putText(map_canvas, f"Position: ({x:.2f}, {y:.2f})", (10, 80),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(map_canvas, f"Yaw: {math.degrees(yaw):.1f}deg", (10, 115),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(map_canvas, f"Conf: {confidence*100:.1f}%", (10, 150),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(map_canvas, f"Matched KF: #{matched_kf['id']}", (10, 185),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(map_canvas, f"Matches: {len(matches)}", (10, 220),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Legend
    y_legend = size - 125
    cv2.putText(map_canvas, "ORANGE = Path", (10, y_legend),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(map_canvas, "PINK = You Are Here", (10, y_legend + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, PINK, 2)
    cv2.putText(map_canvas, "BLUE = Your Keyframe", (10, y_legend + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(map_canvas, "GREEN = Matched Keyframe", (10, y_legend + 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(map_canvas, "GRAY = Other Keyframes", (10, y_legend + 100),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
    
    # Test image with matches
    test_vis = test_image.copy()
    for m in matches:
        pt = kp[m.queryIdx].pt
        cv2.circle(test_vis, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
    
    test_vis = cv2.resize(test_vis, (640, 480))
    cv2.putText(test_vis, "TEST IMAGE", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, PINK, 2)
    cv2.putText(test_vis, f"Matched to Keyframe #{matched_kf['id']}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("Test Image", test_vis)
    cv2.imshow("Map - Keyframe Localization", map_canvas)
    
    print("\n" + "="*70)
    print("📍 SUCCESS!")
    print("="*70)
    print("🟠 ORANGE = Path")
    print("💗 PINK = Your estimated position")
    print("🔵 BLUE = Your keyframe indicator")
    print("🟢 GREEN = Matched keyframe")
    print("⚪ GRAY = Other keyframes")
    print("\n⌨️  Press any key...")
    print("="*70)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"pose_keyframe_{timestamp}.png", map_canvas)
    print(f"\n✅ Saved: pose_keyframe_{timestamp}.png")


def estimate_pose_advanced(img_path, map_data_path, trajectory_from_map=None):
    """Fallback: Advanced pose estimation without keyframes"""
    print("\n🔄 Using fallback feature-based localization...")
    # Original implementation from your code
    pass


# ========== MENU ==========

def main_menu():
    print("\n" + "="*70)
    print("🗺️  SLAM with Keyframes")
    print("="*70)
    print("1. Build Map")
    print("2. Test Pose")
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
    
    rot_sens = input("Rotation [3.0]: ").strip()
    rotation_sensitivity = float(rot_sens) if rot_sens else 3.0
    
    kf_trans = input("Keyframe translation threshold [0.15m]: ").strip()
    keyframe_trans = float(kf_trans) if kf_trans else 0.15
    
    kf_rot = input("Keyframe rotation threshold [15deg]: ").strip()
    keyframe_rot = float(kf_rot) if kf_rot else 15.0
    
    slam = EnhancedSLAM(
        fx=600.0, fy=600.0, cx=320.0, cy=240.0,
        min_features_per_frame=50,
        max_features_per_frame=150,
        rotation_sensitivity=rotation_sensitivity,
        feature_lifetime_frames=50,
        max_depth_threshold=3.0,
        spatial_grid_size=0.10,
        keyframe_translation_threshold=keyframe_trans,
        keyframe_rotation_threshold=keyframe_rot
    )
    
    if choice == '0':
        slam.run_slam(source=None, use_rover_stream=True,
                     rover_url="http://10.47.11.127:8080/video_feed")
    elif choice == '1':
        slam.run_slam(source=0)
    elif choice == '2':
        path = input("Video: ").strip()
        if os.path.exists(path):
            slam.run_slam(source=path)


def estimate_menu():
    print("\n🔍 TEST POSE")
    
    import glob
    maps = sorted(glob.glob("map_*.pkl"), reverse=True)
    
    if not maps:
        print("❌ No maps!")
        return
    
    print(f"✅ Using: {maps[0]}")
    
    img = input("📸 Image: ").strip()
    
    if os.path.exists(img):
        estimate_pose_with_keyframes(img, maps[0])
    else:
        print("❌ Not found")


if __name__ == "__main__":
    main_menu()