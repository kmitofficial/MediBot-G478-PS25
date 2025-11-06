import cv2
import torch
import numpy as np
import time
import os
import pickle
import requests
from dataclasses import dataclass
from typing import List, Tuple, Optional

# =============== Configuration ===============
@dataclass
class SLAMConfig:
    """ORB-SLAM3 inspired configuration"""
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0
    
    n_features: int = 2000
    scale_factor: float = 1.2
    n_levels: int = 8
    
    max_depth: float = 50.0
    min_depth: float = 0.1
    
    min_matches: int = 20
    ransac_threshold: float = 0.01
    
    loop_closure_threshold: float = 0.85
    min_loop_closure_distance: int = 100


# =============== Load Depth Scale ===============
try:
    DEPTH_SCALE = np.load("depth_scale_factor.npy").item()
except Exception:
    DEPTH_SCALE = 5.0
    print(f"⚠️  Using default depth scale: {DEPTH_SCALE}")


# =============== MiDaS Depth Estimator ===============
class DepthEstimator:
    """MiDaS depth estimation wrapper"""
    
    def __init__(self):
        print("🔧 Initializing MiDaS depth estimator...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        
        self.midas.to(self.device).eval()
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CPU mode")
    
    def estimate(self, frame_bgr):
        """Estimate depth map from BGR frame"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        return depth_map
    
    def inverse_depth_to_metric(self, midas_depth):
        """Convert MiDaS inverse depth to metric depth"""
        metric_depth = np.zeros_like(midas_depth, dtype=np.float32)
        valid_mask = midas_depth > 1e-3
        metric_depth[valid_mask] = DEPTH_SCALE / (midas_depth[valid_mask] + 1e-6)
        return metric_depth


# =============== MapPoint (3D Point in Map) ===============
@dataclass
class MapPoint:
    """3D point in the map"""
    x: float
    y: float
    z: float
    descriptor: np.ndarray
    observations: int = 0
    frame_id: int = 0
    is_keyframe_point: bool = False


# =============== KeyFrame ===============
class KeyFrame:
    """ORB-SLAM3 inspired KeyFrame"""
    
    def __init__(self, frame_id, pose, keypoints, descriptors, depth_map, image_gray):
        self.frame_id = frame_id
        self.pose = pose.copy()  # [x, y, z, yaw, pitch, roll]
        self.keypoints_data = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response) for kp in keypoints]
        self.descriptors = descriptors
        self.depth_map = depth_map
        self.image_gray = image_gray.copy()
        self.thumbnail = cv2.resize(image_gray, (160, 120))  # For localization matching


# =============== Rover Stream Capture ===============
class RoverStreamCapture:
    """HTTP MJPEG stream capture for rover camera"""
    
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
            print(f"✅ Connected to rover stream: {self.stream_url}")
        except Exception as e:
            print(f"❌ Failed to connect to rover: {e}")
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


# =============== ORB Feature Extractor ===============
class ORBExtractor:
    """ORB-SLAM3 style feature extraction"""
    
    def __init__(self, config: SLAMConfig):
        self.config = config
        self.orb = cv2.ORB_create(
            nfeatures=config.n_features,
            scaleFactor=config.scale_factor,
            nlevels=config.n_levels,
            edgeThreshold=19,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    def extract(self, image_gray):
        """Extract ORB features from image"""
        enhanced = self.clahe.apply(image_gray)
        keypoints, descriptors = self.orb.detectAndCompute(enhanced, None)
        return keypoints, descriptors


# =============== ORB-SLAM3 System ===============
class ORBSLAM3:
    """Complete ORB-SLAM3 inspired monocular SLAM system"""
    
    def __init__(self, config: SLAMConfig):
        self.config = config
        self.depth_estimator = DepthEstimator()
        self.orb_extractor = ORBExtractor(config)
        
        self.K = np.array([
            [config.fx, 0, config.cx],
            [0, config.fy, config.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Map data - storing continuously
        self.map_points: List[MapPoint] = []
        self.keyframes: List[KeyFrame] = []
        self.current_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.trajectory = []
        
        # Temporary feature points for visualization (disappearing)
        self.temp_feature_points = []  # Only for current frame visualization
        
        self.is_initialized = False
        self.frame_count = 0
        self.last_keyframe_id = -1
        
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_depth = None
        
        self.stats = {
            'features_detected': 0,
            'features_tracked': 0,
            'map_points': 0,
            'keyframes': 0,
            'fps': 0.0
        }
        
        print("✅ ORB-SLAM3 System initialized")
    
    def initialize(self, frame_gray, depth_map):
        """Initialize the map with first frame"""
        print("🔧 Initializing map...")
        
        keypoints, descriptors = self.orb_extractor.extract(frame_gray)
        
        if descriptors is None or len(keypoints) < 100:
            return False
        
        metric_depth = self.depth_estimator.inverse_depth_to_metric(depth_map)
        
        # Create initial map points (stored permanently)
        for i, kp in enumerate(keypoints):
            u, v = int(kp.pt[0]), int(kp.pt[1])
            
            if 0 <= u < frame_gray.shape[1] and 0 <= v < frame_gray.shape[0]:
                depth = metric_depth[v, u]
                
                if self.config.min_depth < depth < self.config.max_depth:
                    x, y, z = self._backproject_pixel(u, v, depth, self.current_pose)
                    
                    map_point = MapPoint(
                        x=x, y=y, z=z,
                        descriptor=descriptors[i],
                        observations=1,
                        frame_id=self.frame_count,
                        is_keyframe_point=True
                    )
                    self.map_points.append(map_point)
        
        # Create first keyframe
        keyframe = KeyFrame(
            frame_id=self.frame_count,
            pose=self.current_pose,
            keypoints=keypoints,
            descriptors=descriptors,
            depth_map=metric_depth,
            image_gray=frame_gray
        )
        self.keyframes.append(keyframe)
        self.last_keyframe_id = self.frame_count
        
        self.prev_frame = frame_gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_depth = metric_depth
        
        self.is_initialized = True
        self.trajectory.append(self.current_pose[:2].copy())  # Store only x,y for 2D
        
        print(f"✅ Map initialized with {len(self.map_points)} points")
        return True
    
    def _backproject_pixel(self, u, v, depth, pose):
        """Backproject pixel to 3D world coordinates"""
        X_cam = (u - self.config.cx) * depth / self.config.fx
        Y_cam = (v - self.config.cy) * depth / self.config.fy
        Z_cam = depth
        
        yaw = pose[3]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        x_world = pose[0] + (cos_yaw * Z_cam + sin_yaw * X_cam)
        y_world = pose[1] + (sin_yaw * Z_cam - cos_yaw * X_cam)
        z_world = pose[2] + Y_cam
        
        return x_world, y_world, z_world
    
    def track(self, frame_gray, depth_map):
        """Track camera pose using current frame"""
        if not self.is_initialized:
            return self.initialize(frame_gray, depth_map)
        
        keypoints, descriptors = self.orb_extractor.extract(frame_gray)
        
        if descriptors is None or len(keypoints) < 10:
            return False
        
        # Update temporary feature points for visualization (disappearing)
        metric_depth = self.depth_estimator.inverse_depth_to_metric(depth_map)
        self.temp_feature_points = []
        for i, kp in enumerate(keypoints[:100]):  # Show only 100 for performance
            u, v = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= u < frame_gray.shape[1] and 0 <= v < frame_gray.shape[0]:
                depth = metric_depth[v, u]
                if self.config.min_depth < depth < self.config.max_depth:
                    x, y, z = self._backproject_pixel(u, v, depth, self.current_pose)
                    self.temp_feature_points.append((x, y))
        
        matches = self._match_features(self.prev_descriptors, descriptors)
        
        if len(matches) < self.config.min_matches:
            print(f"⚠️  Tracking lost: only {len(matches)} matches")
            return False
        
        delta_pose = self._estimate_motion(
            self.prev_keypoints, keypoints, matches,
            self.prev_depth, depth_map
        )
        
        self.current_pose += delta_pose
        self.trajectory.append(self.current_pose[:2].copy())
        
        if self._should_insert_keyframe(matches):
            self._insert_keyframe(frame_gray, keypoints, descriptors, depth_map)
        
        self.prev_frame = frame_gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_depth = metric_depth
        
        self.stats['features_detected'] = len(keypoints)
        self.stats['features_tracked'] = len(matches)
        
        return True
    
    def _match_features(self, desc1, desc2):
        """Match features between two frames"""
        if desc1 is None or desc2 is None:
            return []
        
        try:
            matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            return good_matches
        except:
            return []
    
    def _estimate_motion(self, prev_kp, curr_kp, matches, prev_depth, curr_depth_map):
        """Estimate camera motion between frames"""
        if len(matches) < 5:
            return np.zeros(6)
        
        prev_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
        curr_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches])
        
        flow = curr_pts - prev_pts
        median_flow = np.median(flow, axis=0)
        
        delta_pose = np.zeros(6)
        delta_yaw = median_flow[0] / self.config.fx * 0.1
        delta_pose[3] = delta_yaw
        
        if median_flow[1] > 0.5:
            delta_pose[0] = 0.01 * np.cos(self.current_pose[3])
            delta_pose[1] = 0.01 * np.sin(self.current_pose[3])
        
        return delta_pose
    
    def _should_insert_keyframe(self, matches):
        """Decide if we need a new keyframe"""
        frames_since_last_kf = self.frame_count - self.last_keyframe_id
        return frames_since_last_kf > 10 or len(matches) < 50
    
    def _insert_keyframe(self, frame_gray, keypoints, descriptors, depth_map):
        """Insert new keyframe and expand map"""
        metric_depth = self.depth_estimator.inverse_depth_to_metric(depth_map)
        
        keyframe = KeyFrame(
            frame_id=self.frame_count,
            pose=self.current_pose.copy(),
            keypoints=keypoints,
            descriptors=descriptors,
            depth_map=metric_depth,
            image_gray=frame_gray
        )
        
        # Add new map points (stored permanently)
        new_points = 0
        for i, kp in enumerate(keypoints):
            u, v = int(kp.pt[0]), int(kp.pt[1])
            
            if 0 <= u < frame_gray.shape[1] and 0 <= v < frame_gray.shape[0]:
                depth = metric_depth[v, u]
                
                if self.config.min_depth < depth < self.config.max_depth:
                    x, y, z = self._backproject_pixel(u, v, depth, self.current_pose)
                    
                    if not self._point_exists_nearby(x, y, z, threshold=0.1):
                        map_point = MapPoint(
                            x=x, y=y, z=z,
                            descriptor=descriptors[i],
                            observations=1,
                            frame_id=self.frame_count,
                            is_keyframe_point=True
                        )
                        self.map_points.append(map_point)
                        new_points += 1
        
        self.keyframes.append(keyframe)
        self.last_keyframe_id = self.frame_count
        
        self.stats['map_points'] = len(self.map_points)
        self.stats['keyframes'] = len(self.keyframes)
        
        if new_points > 0:
            print(f"🔑 Keyframe {len(self.keyframes)}: +{new_points} points (total: {len(self.map_points)})")
    
    def _point_exists_nearby(self, x, y, z, threshold=0.1):
        """Check if a map point exists nearby"""
        for mp in self.map_points[-1000:]:
            dist = np.sqrt((mp.x - x)**2 + (mp.y - y)**2 + (mp.z - z)**2)
            if dist < threshold:
                return True
        return False
    
    def process_frame(self, frame_bgr):
        """Process a single frame"""
        self.frame_count += 1
        
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        depth_map = self.depth_estimator.estimate(frame_bgr)
        
        if not self.is_initialized:
            success = self.initialize(frame_gray, depth_map)
        else:
            success = self.track(frame_gray, depth_map)
        
        return success, frame_gray, depth_map
    
    def save_map(self, filename="slam_map.pkl"):
        """Save map to file"""
        print(f"\n💾 Saving map to {filename}...")
        
        serializable_keyframes = []
        for kf in self.keyframes:
            kf_data = {
                'frame_id': kf.frame_id,
                'pose': kf.pose,
                'keypoints_data': kf.keypoints_data,
                'descriptors': kf.descriptors,
                'thumbnail': kf.thumbnail
            }
            serializable_keyframes.append(kf_data)
        
        data = {
            'map_points': self.map_points,
            'keyframes': serializable_keyframes,
            'trajectory': self.trajectory,
            'config': self.config,
            'current_pose': self.current_pose
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✅ Saved {len(self.map_points)} map points, {len(self.keyframes)} keyframes")
        return filename


# =============== 2D Top-Down Map Visualizer ===============
class TopDownMapVisualizer:
    """2D top-down map visualization"""
    
    def __init__(self, width=1200, height=900):
        self.width = width
        self.height = height
        self.scale = 60  # pixels per meter
        self.center_x = width // 2
        self.center_y = height // 2
        
        # Colors
        self.COLOR_BACKGROUND = (30, 30, 30)
        self.COLOR_GRID = (60, 60, 60)
        self.COLOR_KEYFRAME = (255, 150, 0)  # Blue for keyframes
        self.COLOR_TEMP_POINT = (100, 255, 100)  # Green for temporary features
        self.COLOR_TRAJECTORY = (0, 165, 255)  # Orange for path
        self.COLOR_CURRENT = (255, 0, 255)  # Magenta for current position
        
        cv2.namedWindow("2D Top-Down Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("2D Top-Down Map", width, height)
    
    def world_to_screen(self, x, y, origin_x=0, origin_y=0):
        """Convert world coordinates to screen coordinates"""
        dx = (x - origin_x) * self.scale
        dy = (y - origin_y) * self.scale
        
        sx = int(self.center_x + dx)
        sy = int(self.center_y - dy)  # Flip Y axis
        
        return sx, sy
    
    def draw_grid(self, canvas):
        """Draw grid lines"""
        grid_spacing = 1.0  # meters
        pixel_spacing = int(grid_spacing * self.scale)
        
        # Vertical lines
        for x in range(0, self.width, pixel_spacing):
            cv2.line(canvas, (x, 0), (x, self.height), self.COLOR_GRID, 1)
        
        # Horizontal lines
        for y in range(0, self.height, pixel_spacing):
            cv2.line(canvas, (0, y), (self.width, y), self.COLOR_GRID, 1)
        
        # Center axes (thicker)
        cv2.line(canvas, (self.center_x, 0), (self.center_x, self.height), (80, 80, 80), 2)
        cv2.line(canvas, (0, self.center_y), (self.width, self.center_y), (80, 80, 80), 2)
    
    def update(self, slam):
        """Update and display the map"""
        canvas = np.full((self.height, self.width, 3), self.COLOR_BACKGROUND, dtype=np.uint8)
        
        # Draw grid
        self.draw_grid(canvas)
        
        # Get current position for centering
        origin_x, origin_y = 0, 0
        if len(slam.trajectory) > 0:
            origin_x, origin_y = slam.current_pose[0], slam.current_pose[1]
        
        # Draw trajectory (orange path)
        if len(slam.trajectory) > 1:
            for i in range(len(slam.trajectory) - 1):
                p1 = self.world_to_screen(slam.trajectory[i][0], slam.trajectory[i][1], origin_x, origin_y)
                p2 = self.world_to_screen(slam.trajectory[i+1][0], slam.trajectory[i+1][1], origin_x, origin_y)
                
                if (0 <= p1[0] < self.width and 0 <= p1[1] < self.height and
                    0 <= p2[0] < self.width and 0 <= p2[1] < self.height):
                    cv2.line(canvas, p1, p2, self.COLOR_TRAJECTORY, 3)
        
        # Draw temporary feature points (disappearing green dots)
        for x, y in slam.temp_feature_points:
            sx, sy = self.world_to_screen(x, y, origin_x, origin_y)
            if 0 <= sx < self.width and 0 <= sy < self.height:
                cv2.circle(canvas, (sx, sy), 2, self.COLOR_TEMP_POINT, -1)
        
        # Draw keyframes (fixed blue circles)
        for kf in slam.keyframes:
            kf_x, kf_y = kf.pose[0], kf.pose[1]
            sx, sy = self.world_to_screen(kf_x, kf_y, origin_x, origin_y)
            
            if 0 <= sx < self.width and 0 <= sy < self.height:
                cv2.circle(canvas, (sx, sy), 8, self.COLOR_KEYFRAME, -1)
                cv2.circle(canvas, (sx, sy), 10, self.COLOR_KEYFRAME, 2)
                
                # Draw keyframe ID
                cv2.putText(canvas, f"KF{kf.frame_id}", (sx + 12, sy - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_KEYFRAME, 1)
        
        # Draw current position (magenta)
        current_pos = self.world_to_screen(origin_x, origin_y, origin_x, origin_y)
        cv2.circle(canvas, current_pos, 12, self.COLOR_CURRENT, -1)
        cv2.circle(canvas, current_pos, 15, self.COLOR_CURRENT, 3)
        
        # Draw heading arrow
        yaw = slam.current_pose[3]
        arrow_len = 30
        arrow_end = (
            int(current_pos[0] + arrow_len * np.cos(yaw)),
            int(current_pos[1] - arrow_len * np.sin(yaw))
        )
        cv2.arrowedLine(canvas, current_pos, arrow_end, self.COLOR_CURRENT, 3, tipLength=0.3)
        
        # Draw legend and stats
        self._draw_legend(canvas, slam)
        
        cv2.imshow("2D Top-Down Map", canvas)
    
    def _draw_legend(self, canvas, slam):
        """Draw legend and statistics"""
        y = 30
        
        # Title
        cv2.putText(canvas, "2D TOP-DOWN MAP", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        y += 40
        
        # Stats
        distance = np.linalg.norm(slam.current_pose[:2])
        
        cv2.putText(canvas, f"Frame: {slam.frame_count}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        
        cv2.putText(canvas, f"Map Points: {len(slam.map_points)}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        
        cv2.putText(canvas, f"Keyframes: {len(slam.keyframes)}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        
        cv2.putText(canvas, f"Distance: {distance:.2f}m", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 25
        
        cv2.putText(canvas, f"FPS: {slam.stats['fps']:.1f}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 40
        
        # Legend
        legend_y = self.height - 120
        cv2.putText(canvas, "LEGEND:", (20, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 20
        
        cv2.circle(canvas, (30, legend_y), 6, self.COLOR_KEYFRAME, -1)
        cv2.putText(canvas, "Keyframes", (45, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        legend_y += 25
        
        cv2.circle(canvas, (30, legend_y), 3, self.COLOR_TEMP_POINT, -1)
        cv2.putText(canvas, "Feature Points", (45, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        legend_y += 25
        
        cv2.line(canvas, (25, legend_y), (35, legend_y), self.COLOR_TRAJECTORY, 3)
        cv2.putText(canvas, "Trajectory", (45, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        legend_y += 25
        
        cv2.circle(canvas, (30, legend_y), 6, self.COLOR_CURRENT, -1)
        cv2.putText(canvas, "Current Position", (45, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def close(self):
        cv2.destroyWindow("2D Top-Down Map")


# =============== Camera View ===============
def draw_camera_view(frame_bgr, slam, fps=0.0):
    """Draw camera view with ORB features"""
    vis = frame_bgr.copy()
    
    if slam.prev_keypoints is not None:
        for kp in slam.prev_keypoints[:200]:
            pt = (int(kp.pt[0]), int(kp.pt[1]))
            cv2.circle(vis, pt, 3, (0, 255, 0), -1)
    
    h, w = vis.shape[:2]
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (400, 150), (0, 0, 0), -1)
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    
    distance = np.linalg.norm(slam.current_pose[:2])
    
    y = 30
    cv2.putText(vis, f"Frame: {slam.frame_count}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 30
    cv2.putText(vis, f"Keyframes: {len(slam.keyframes)}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 30
    cv2.putText(vis, f"Features: {slam.stats['features_detected']}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 30
    cv2.putText(vis, f"Distance: {distance:.2f}m", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 30
    cv2.putText(vis, f"FPS: {fps:.1f}", (10, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    status = "TRACKING" if slam.is_initialized else "INITIALIZING"
    status_color = (0, 255, 0) if slam.is_initialized else (0, 165, 255)
    cv2.putText(vis, status, (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    return vis


# =============== Localization with Keyframe Matching ===============
def localize_image(image_path, map_path):
    """Localize a query image and find best matching keyframe"""
    print("\n" + "="*70)
    print("📍 LOCALIZATION MODE")
    print("="*70)
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    query_img = cv2.imread(image_path)
    if query_img is None:
        print(f"❌ Failed to load image: {image_path}")
        return
    
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    print(f"✅ Query image loaded: {query_img.shape}")
    
    if not os.path.exists(map_path):
        print(f"❌ Map not found: {map_path}")
        return
    
    print(f"📂 Loading map from {map_path}...")
    with open(map_path, 'rb') as f:
        map_data = pickle.load(f)
    
    map_points = map_data['map_points']
    keyframes_data = map_data['keyframes']
    trajectory = map_data['trajectory']
    config = map_data.get('config', SLAMConfig())
    
    print(f"✅ Map loaded: {len(map_points)} points, {len(keyframes_data)} keyframes")
    
    # Extract features from query
    orb_extractor = ORBExtractor(config)
    query_kp, query_desc = orb_extractor.extract(query_gray)
    
    if query_desc is None or len(query_kp) == 0:
        print("❌ No features detected in query image")
        return
    
    print(f"✅ Query features: {len(query_kp)}")
    
    # Match against all keyframes to find best match
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    best_keyframe_idx = -1
    best_match_count = 0
    best_matches = []
    
    print("\n🔍 Matching against keyframes...")
    for idx, kf_data in enumerate(keyframes_data):
        kf_desc = kf_data['descriptors']
        
        if kf_desc is None:
            continue
        
        matches = bf.knnMatch(query_desc, kf_desc, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        print(f"  KF {kf_data['frame_id']}: {len(good_matches)} matches")
        
        if len(good_matches) > best_match_count:
            best_match_count = len(good_matches)
            best_keyframe_idx = idx
            best_matches = good_matches
    
    if best_keyframe_idx == -1 or best_match_count < 8:
        print(f"\n❌ No good keyframe match found (best: {best_match_count} matches)")
        return
    
    best_kf = keyframes_data[best_keyframe_idx]
    print(f"\n✅ Best match: Keyframe {best_kf['frame_id']} with {best_match_count} matches")
    
    # Estimate position using map points
    map_descriptors = np.array([mp.descriptor for mp in map_points])
    matches_map = bf.knnMatch(query_desc, map_descriptors, k=2)
    
    good_matches_map = []
    for match_pair in matches_map:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches_map.append(m)
    
    if len(good_matches_map) < 8:
        print(f"❌ Insufficient map point matches: {len(good_matches_map)}")
        return
    
    matched_points = [map_points[m.trainIdx] for m in good_matches_map[:100]]
    
    xs = [mp.x for mp in matched_points]
    ys = [mp.y for mp in matched_points]
    
    x_est = np.median(xs)
    y_est = np.median(ys)
    
    # Refine with inliers
    distances = [np.sqrt((mp.x - x_est)**2 + (mp.y - y_est)**2) for mp in matched_points]
    threshold = np.percentile(distances, 75)
    inliers = [mp for mp, d in zip(matched_points, distances) if d < threshold]
    
    if len(inliers) >= 8:
        x_est = np.median([mp.x for mp in inliers])
        y_est = np.median([mp.y for mp in inliers])
    
    avg_match_distance = np.mean([m.distance for m in best_matches])
    confidence = max(0.0, 1.0 - (avg_match_distance / 100.0))
    
    print("\n" + "="*70)
    print("📍 LOCALIZATION RESULT")
    print("="*70)
    print(f"Best Matching Keyframe: {best_kf['frame_id']}")
    print(f"Keyframe Position: ({best_kf['pose'][0]:.3f}, {best_kf['pose'][1]:.3f})")
    print(f"Estimated Position: ({x_est:.3f}, {y_est:.3f})")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"Matches: {best_match_count} (keyframe) / {len(inliers)} (map)")
    print("="*70)
    
    # Visualize
    visualize_localization(query_img, best_kf, trajectory, x_est, y_est, confidence, best_match_count)


def visualize_localization(query_img, best_kf, trajectory, x_est, y_est, confidence, match_count):
    """Visualize localization result"""
    
    # Show query image
    vis_query = query_img.copy()
    h, w = vis_query.shape[:2]
    overlay = vis_query.copy()
    cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
    vis_query = cv2.addWeighted(vis_query, 0.7, overlay, 0.3, 0)
    
    cv2.putText(vis_query, "QUERY IMAGE", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 105, 180), 2)
    cv2.putText(vis_query, f"Best KF: {best_kf['frame_id']} | Matches: {match_count} | Conf: {confidence*100:.1f}%", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis_query, f"Position: ({x_est:.2f}, {y_est:.2f})", 
               (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Draw 2D map
    map_size = 900
    scale = 50
    map_canvas = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    map_canvas[:] = (30, 30, 30)
    center_x, center_y = map_size // 2, map_size // 2
    
    def project(x, y):
        dx = (x - x_est) * scale
        dy = (y - y_est) * scale
        sx = int(center_x + dx)
        sy = int(center_y - dy)
        return sx, sy
    
    # Draw trajectory
    if len(trajectory) > 1:
        for i in range(len(trajectory) - 1):
            p1 = project(trajectory[i][0], trajectory[i][1])
            p2 = project(trajectory[i+1][0], trajectory[i+1][1])
            
            if (0 <= p1[0] < map_size and 0 <= p1[1] < map_size and
                0 <= p2[0] < map_size and 0 <= p2[1] < map_size):
                cv2.line(map_canvas, p1, p2, (0, 165, 255), 3)
    
    # Draw best matching keyframe
    kf_pos = project(best_kf['pose'][0], best_kf['pose'][1])
    if 0 <= kf_pos[0] < map_size and 0 <= kf_pos[1] < map_size:
        cv2.circle(map_canvas, kf_pos, 12, (255, 150, 0), -1)
        cv2.circle(map_canvas, kf_pos, 15, (255, 150, 0), 3)
        cv2.putText(map_canvas, f"KF{best_kf['frame_id']}", (kf_pos[0] + 18, kf_pos[1] - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
    
    # Draw estimated position
    PINK = (255, 192, 203)
    est_pos = project(x_est, y_est)
    cv2.circle(map_canvas, est_pos, 15, PINK, -1)
    cv2.circle(map_canvas, est_pos, 20, PINK, 3)
    
    # Title and info
    cv2.putText(map_canvas, "LOCALIZATION RESULT", (10, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, PINK, 3)
    cv2.putText(map_canvas, f"Matched Keyframe: {best_kf['frame_id']}", (10, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(map_canvas, f"Confidence: {confidence*100:.1f}%", (10, 105),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Show matched keyframe thumbnail
    if 'thumbnail' in best_kf and best_kf['thumbnail'] is not None:
        thumb = best_kf['thumbnail']
        thumb_color = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        th, tw = thumb_color.shape[:2]
        map_canvas[map_size-th-10:map_size-10, 10:10+tw] = thumb_color
        cv2.rectangle(map_canvas, (10, map_size-th-10), (10+tw, map_size-10), (255, 150, 0), 2)
        cv2.putText(map_canvas, f"Matched KF {best_kf['frame_id']}", (10, map_size-th-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
    
    cv2.imshow("Query Image", vis_query)
    cv2.imshow("Localization Map", map_canvas)
    
    print("\n⌨️  Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"localization_{timestamp}.png", map_canvas)
    print(f"✅ Result saved")


# =============== Main Building Mode ===============
def build_map_mode():
    """Build 2D map from video or rover stream"""
    print("\n" + "="*70)
    print("🗺️  MAP BUILDING MODE")
    print("="*70)
    print("1. Video file")
    print("2. Rover camera stream")
    print("="*70)
    
    choice = input("Select source [1/2]: ").strip()
    
    if choice == '1':
        video_path = input("📹 Enter video path: ").strip()
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return
        cap = cv2.VideoCapture(video_path)
    elif choice == '2':
        rover_url = input("🤖 Rover stream URL [http://10.47.11.127:8080/video_feed]: ").strip()
        if not rover_url:
            rover_url = "http://10.47.11.127:8080/video_feed"
        cap = RoverStreamCapture(rover_url)
    else:
        print("❌ Invalid choice")
        return
    
    if not cap.isOpened():
        print("❌ Failed to open video source")
        return
    
    config = SLAMConfig()
    slam = ORBSLAM3(config)
    
    visualizer = TopDownMapVisualizer()
    
    print("\n" + "="*70)
    print("🎬 BUILDING MAP...")
    print("="*70)
    print("⌨️  [S] - Save map | [ESC] - Quit")
    print("="*70 + "\n")
    
    fps = 0.0
    prev_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  End of video")
                break
            
            frame = cv2.resize(frame, (640, 480))
            
            success, frame_gray, depth_map = slam.process_frame(frame)
            
            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else 1.0 / dt
            slam.stats['fps'] = fps
            
            camera_view = draw_camera_view(frame, slam, fps=fps)
            cv2.imshow("Camera View", camera_view)
            
            visualizer.update(slam)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('s') or key == ord('S'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                map_filename = f"slam_map_{timestamp}.pkl"
                slam.save_map(map_filename)
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        visualizer.close()
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        map_filename = f"slam_map_{timestamp}.pkl"
        slam.save_map(map_filename)
        
        print(f"\n✅ Map saved: {map_filename}")


# =============== Main Localization Mode ===============
def localization_mode():
    """Localization mode"""
    print("\n" + "="*70)
    print("📍 LOCALIZATION MODE")
    print("="*70)
    
    import glob
    map_files = sorted(glob.glob("slam_map_*.pkl"), reverse=True)
    
    if not map_files:
        print("❌ No maps found! Build a map first")
        return
    
    print(f"\n✅ Found {len(map_files)} map(s)")
    print(f"📂 Using: {map_files[0]}")
    
    image_path = input("\n📸 Query image path: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ Not found: {image_path}")
        return
    
    localize_image(image_path, map_files[0])


# =============== Main Menu ===============
def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🎯 ORB-SLAM3 Monocular SLAM")
    print("   2D Top-Down Visualization")
    print("="*70)
    print("1. Build Map")
    print("0. Localization")
    print("="*70)
    
    choice = input("Select [1/0]: ").strip()
    
    if choice == '1':
        build_map_mode()
    elif choice == '0':
        localization_mode()
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()