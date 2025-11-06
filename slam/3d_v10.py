# lookahead_slam_v11_with_prediction.py
# Enhanced SLAM with FORWARD-ONLY motion and advanced feature detection + Image Prediction Module
# Modified to support Rover IP camera stream

import cv2
import torch
import math
import numpy as np
import time
import os
import pickle
import requests
from collections import deque

# =============== Load Scaled Depth Factor ===============
try:
    DEPTH_SCALE = np.load("depth_scale_factor.npy").item()
    print(f"✅ Loaded depth scale factor: {DEPTH_SCALE:.4f}")
    print(f"💡 Real distance = {DEPTH_SCALE:.2f} / midas_value\n")
except Exception as e:
    print(f"⚠️ Could not load depth_scale_factor.npy, using default scale")
    DEPTH_SCALE = 5.0  # Default fallback

# =============== MiDaS Depth Model ===============
print("Loading MiDaS depth model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()
print(f"MiDaS ready on {device}\n")


class RoverStreamCapture:
    """
    Custom video capture class for Rover IP camera stream
    Compatible with cv2.VideoCapture interface
    """

    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.resp = None
        self.bytes_buf = b""
        self.is_opened = False
        self._connect()

    def _connect(self):
        """Connect to the rover stream"""
        try:
            self.resp = requests.get(self.stream_url, stream=True, timeout=5)
            if self.resp.status_code == 200:
                self.is_opened = True
                print(f"✅ Connected to rover stream: {self.stream_url}")
            else:
                print(f"❌ Failed to connect: HTTP {self.resp.status_code}")
                self.is_opened = False
        except Exception as e:
            print(f"❌ Connection error: {e}")
            self.is_opened = False

    def isOpened(self):
        """Check if stream is opened"""
        return self.is_opened

    def read(self):
        """Read a frame from the stream"""
        if not self.is_opened:
            return False, None

        try:
            # Read chunks until we get a complete JPEG frame
            for chunk in self.resp.iter_content(chunk_size=1024):
                self.bytes_buf += chunk

                # Find JPEG boundaries
                a = self.bytes_buf.find(b'\xff\xd8')  # JPEG start
                b = self.bytes_buf.find(b'\xff\xd9')  # JPEG end

                if a != -1 and b != -1:
                    jpg = self.bytes_buf[a:b+2]
                    self.bytes_buf = self.bytes_buf[b+2:]

                    # Decode JPEG
                    img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                    if img is not None:
                        return True, img

            return False, None
        except Exception as e:
            print(f"❌ Error reading frame: {e}")
            self.is_opened = False
            return False, None

    def release(self):
        """Release the stream"""
        if self.resp is not None:
            self.resp.close()
        self.is_opened = False
        print("🛑 Rover stream closed.")


class LookaheadSLAM:
    """
    FORWARD-ONLY SLAM system with enhanced feature detection
    Uses CLAHE preprocessing + FAST only for fast feature detection
    """

    def __init__(
        self,
        fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        lookahead_height_ratio=0.9,
        voxel_size=0.08,
        landmark_update_rate=0.20,
        max_landmarks=50000,
        max_features_per_frame=200,
        motion_smooth_alpha=0.40,
        yaw_smooth_alpha=0.40,
        rotation_threshold=0.02,
        min_depth_change=0.01,
        max_depth_change=0.05
    ):
        # Camera intrinsics
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

        # Lookahead configuration
        self.lookahead_height_ratio = lookahead_height_ratio
        self.lookahead_pixel = None
        self.lookahead_depth_prev = None
        self.lookahead_depth_curr = None
        self.lookahead_world_pos = None

        # Pose: [x, y, yaw] in WORLD-FIXED coordinates
        self.pose = np.zeros(3, dtype=float)
        self.trajectory = []
        self.path_distance = 0.0
        self.meter_markers = []  # Store positions at each meter mark

        # Motion smoothing
        self.motion_smooth_alpha = motion_smooth_alpha
        self.yaw_smooth_alpha = yaw_smooth_alpha
        self.smoothed_translation = 0.0
        self.smoothed_yaw = 0.0

        # Rotation handling
        self.rotation_threshold = rotation_threshold
        self.is_rotating = False
        self.raw_yaw_delta = 0.0

        # Distance calculation
        self.min_depth_change = min_depth_change

        # Feature tracking for distance calculation
        self.prev_tracked_features = None  # Store previous frame features for tracking
        self.tracked_frame_count = 0  # Counter for processing every 3rd frame

        # Feature-based landmark map
        self.voxel_size = voxel_size
        self.landmark_update_rate = landmark_update_rate
        self.max_landmarks = max_landmarks
        self.landmarks = {}

        # ========== ENHANCED FEATURE DETECTION ==========
        self.max_features_per_frame = max_features_per_frame

        # Ultra-aggressive CLAHE for preprocessing
        self.clahe_ultra = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(3, 3))

        # ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=max_features_per_frame,
            scaleFactor=1.75,
            nlevels=8,
            edgeThreshold=3,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=20,
            fastThreshold=10
        )

        # GFTT (Shi-Tomasi) parameters
        self.gftt_params = {
            'maxCorners': max_features_per_frame,
            'qualityLevel': 0.01,
            'minDistance': 5,
            'blockSize': 3,
            'useHarrisDetector': False,
            'k': 0.04
        }

        # FAST detector
        self.fast_detector = cv2.FastFeatureDetector_create(
            threshold=15,
            nonmaxSuppression=True,
            type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
        )

        # BFMatcher for feature matching
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Previous frame data
        self.prev_gray = None
        self.prev_depth = None
        self.frame_count = 0

        # Visualization data
        self.current_features = []
        self.feature_counts = {'orb': 0, 'gftt': 0, 'fast': 0, 'total': 0}
        self.rotation_magnitude = 0.0
        self.translation_magnitude = 0.0

        # Cached screen center for map
        self._last_top_map_center = None

        # Store last saved map filename
        self.last_map_filename = None

    def _enhance_image(self, gray):
        """
        Enhanced image preprocessing using CLAHE + contrast enhancement + gamma correction
        """
        # Apply ultra-aggressive CLAHE
        ultra_clahe = self.clahe_ultra.apply(gray)

        # Extreme contrast enhancement
        extreme_contrast = cv2.convertScaleAbs(ultra_clahe, alpha=1.8, beta=15)

        # Gamma correction
        gamma = 0.7
        lookUpTable = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
        gamma_corrected = cv2.LUT(extreme_contrast, lookUpTable)

        # Fast denoising with Gaussian blur
        denoised = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)

        return denoised

    def _voxel_key(self, x, y):
        """Hash world coordinates into voxel grid"""
        return (
            int(math.floor(x / self.voxel_size)),
            int(math.floor(y / self.voxel_size))
        )

    def _compute_lookahead_pixel(self, img_height, img_width):
        """Compute the lookahead point on ground plane"""
        u = self.cx
        v = img_height * self.lookahead_height_ratio
        return (int(u), int(v))

    def _midas_to_metric_depth(self, midas_value):
        """Convert MiDaS inverse depth to metric depth"""
        if midas_value < 1e-3:
            return None
        return DEPTH_SCALE / (midas_value + 1e-6)

    def _backproject_to_world(self, u, v, metric_depth, pose):
        """Backproject pixel to world coordinates"""
        if metric_depth is None or metric_depth <= 0:
            return None

        # Backproject to camera frame
        X_cam = (u - self.cx) * metric_depth / self.fx
        Y_cam = (v - self.cy) * metric_depth / self.fy
        Z_cam = metric_depth

        # Transform to world frame
        yaw = pose[2]
        c, s = math.cos(yaw), math.sin(yaw)
        x_world = pose[0] + (c * Z_cam + s * X_cam)
        y_world = pose[1] + (s * Z_cam - c * X_cam)

        return (x_world, y_world, Z_cam)

    def _update_landmark(self, x_world, y_world, z_metric, descriptor):
        """Add or update landmark in voxel map"""
        key = self._voxel_key(x_world, y_world)

        if key in self.landmarks:
            lm = self.landmarks[key]
            beta = self.landmark_update_rate
            lm['x'] = (1 - beta) * lm['x'] + beta * x_world
            lm['y'] = (1 - beta) * lm['y'] + beta * y_world
            lm['z'] = (1 - beta) * lm['z'] + beta * z_metric
            lm['descriptor'] = descriptor
            lm['count'] += 1
        else:
            if len(self.landmarks) < self.max_landmarks:
                self.landmarks[key] = {
                    'x': x_world,
                    'y': y_world,
                    'z': z_metric,
                    'descriptor': descriptor,
                    'count': 1
                }

    def _detect_and_build_map(self, gray, depth_map, pose):
        """
        Enhanced feature detection using CLAHE + ORB + GFTT + FAST
        """
        # Enhance image using CLAHE preprocessing
        enhanced = self._enhance_image(gray)

        # Detect features using all three detectors
        kp_orb, desc_orb = self.orb.detectAndCompute(enhanced, None)
        corners_gftt = cv2.goodFeaturesToTrack(enhanced, **self.gftt_params)
        kp_fast = self.fast_detector.detect(enhanced, None)

        # Store feature counts for visualization
        self.feature_counts['orb'] = len(kp_orb) if kp_orb is not None else 0
        self.feature_counts['gftt'] = len(corners_gftt) if corners_gftt is not None else 0
        self.feature_counts['fast'] = len(kp_fast) if kp_fast is not None else 0

        img_height, img_width = gray.shape
        feature_list = []

        # Process ORB features (with descriptors for matching)
        if kp_orb is not None and desc_orb is not None:
            for kp, desc in zip(kp_orb, desc_orb):
                u, v = int(kp.pt[0]), int(kp.pt[1])
                if not (0 <= u < img_width and 0 <= v < img_height):
                    continue

                midas_depth = depth_map[v, u]
                metric_depth = self._midas_to_metric_depth(midas_depth)

                if metric_depth is not None:
                    world_pos = self._backproject_to_world(u, v, metric_depth, pose)
                    if world_pos is not None:
                        x_world, y_world, _ = world_pos
                        self._update_landmark(x_world, y_world, metric_depth, desc)
                        feature_list.append((u, v, 'orb'))

        # Process GFTT features (convert to keypoints)
        if corners_gftt is not None and len(corners_gftt) > 0:
            for corner in corners_gftt:
                u, v = int(corner[0][0]), int(corner[0][1])
                if not (0 <= u < img_width and 0 <= v < img_height):
                    continue

                midas_depth = depth_map[v, u]
                metric_depth = self._midas_to_metric_depth(midas_depth)

                if metric_depth is not None:
                    world_pos = self._backproject_to_world(u, v, metric_depth, pose)
                    if world_pos is not None:
                        x_world, y_world, _ = world_pos
                        # GFTT doesn't have descriptors, use None
                        self._update_landmark(x_world, y_world, metric_depth, None)
                        feature_list.append((u, v, 'gftt'))

        # Process FAST features
        if kp_fast is not None:
            for kp in kp_fast:
                u, v = int(kp.pt[0]), int(kp.pt[1])
                if not (0 <= u < img_width and 0 <= v < img_height):
                    continue

                midas_depth = depth_map[v, u]
                metric_depth = self._midas_to_metric_depth(midas_depth)

                if metric_depth is not None:
                    world_pos = self._backproject_to_world(u, v, metric_depth, pose)
                    if world_pos is not None:
                        x_world, y_world, _ = world_pos
                        # FAST doesn't have descriptors, use None
                        self._update_landmark(x_world, y_world, metric_depth, None)
                        feature_list.append((u, v, 'fast'))

        self.feature_counts['total'] = len(feature_list)
        return feature_list

    def depth_map(self, frame_bgr):
        """Compute MiDaS depth map (inverse depth)"""
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

        depth = pred.cpu().numpy()
        return depth

    def update(self, gray, depth_map):
        """
        Main SLAM update with FORWARD-ONLY motion
        Optimized: Process every 2nd frame for faster performance
        """
        img_height, img_width = gray.shape
        self.frame_count += 1
        self.tracked_frame_count += 1

        # OPTIMIZATION: Process depth estimation and tracking only every 2nd frame
        skip_processing = (self.tracked_frame_count % 2 != 0)

        if skip_processing:
            # Skip heavy processing, just store frame and return
            self.prev_gray = gray
            self.prev_depth = depth_map
            return

        # Compute lookahead pixel position
        self.lookahead_pixel = self._compute_lookahead_pixel(img_height, img_width)
        u_look, v_look = self.lookahead_pixel

        # Get depth at lookahead point
        midas_depth = depth_map[v_look, u_look]
        self.lookahead_depth_curr = self._midas_to_metric_depth(midas_depth)

        if self.lookahead_depth_curr is None:
            self.prev_gray = gray
            self.prev_depth = depth_map
            return

        # Calculate lookahead world position
        self.lookahead_world_pos = self._backproject_to_world(
            u_look, v_look, self.lookahead_depth_curr, self.pose
        )

        # ========== ODOMETRY ==========
        if self.prev_gray is not None and self.lookahead_depth_prev is not None:
            prev_point = np.array([[[u_look, v_look]]], dtype=np.float32)
            next_point, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_point, None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )

            if status is not None and status[0][0] == 1:
                u_next, v_next = next_point[0][0]

                # Rotation estimation
                delta_u = u_next - u_look
                yaw_raw = delta_u / max(1e-6, self.fx)
                yaw_raw *= 6.25
                
                # Apply dead zone to yaw_raw to prevent drift
                yaw_deadzone = 0.001  # Dead zone threshold
                if abs(yaw_raw) < yaw_deadzone:
                    yaw_raw = 0.0
                
                # Smooth yaw - only accumulate when there's actual rotation
                if abs(yaw_raw) > 0:
                    self.smoothed_yaw = (
                        self.yaw_smooth_alpha * yaw_raw +
                        (1 - self.yaw_smooth_alpha) * self.smoothed_yaw
                    )
                else:
                    # Decay smoothed_yaw to zero when no rotation detected
                    self.smoothed_yaw *= 0.5  # Exponential decay
                    if abs(self.smoothed_yaw) < yaw_deadzone:
                        self.smoothed_yaw = 0.0
                
                self.is_rotating = abs(self.smoothed_yaw) > self.rotation_threshold

                # Translation estimation using FEATURE TRACKING
                forward_distance = 0.0

                # Track features from previous frame to current frame for distance calculation
                if self.prev_tracked_features is not None and len(self.prev_tracked_features) > 0:
                    # Track features using optical flow
                    prev_pts = np.array(self.prev_tracked_features, dtype=np.float32).reshape(-1, 1, 2)
                    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                        self.prev_gray, gray, prev_pts, None,
                        winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                    )

                    # Filter good matches
                    if status is not None:
                        good_prev = prev_pts[status.flatten() == 1]
                        good_next = next_pts[status.flatten() == 1]

                        if len(good_prev) >= 5:  # Need at least 5 tracked features
                            # Calculate 3D displacement for each tracked feature
                            displacements_3d = []

                            for (prev_pt, next_pt) in zip(good_prev, good_next):
                                # FIXED: prev_pt and next_pt are [1, 2] arrays, access with [0]
                                u_prev, v_prev = int(prev_pt[0][0]), int(prev_pt[0][1])
                                u_next, v_next = int(next_pt[0][0]), int(next_pt[0][1])

                                # Boundary check
                                if (0 <= u_prev < img_width and 0 <= v_prev < img_height and
                                    0 <= u_next < img_width and 0 <= v_next < img_height):

                                    # Get depths
                                    midas_prev = self.prev_depth[v_prev, u_prev]
                                    midas_next = depth_map[v_next, u_next]

                                    depth_prev = self._midas_to_metric_depth(midas_prev)
                                    depth_next = self._midas_to_metric_depth(midas_next)

                                    if depth_prev is not None and depth_next is not None:
                                        # Backproject to 3D camera coordinates (previous frame)
                                        X_prev = (u_prev - self.cx) * depth_prev / self.fx
                                        Y_prev = (v_prev - self.cy) * depth_prev / self.fy
                                        Z_prev = depth_prev

                                        # Backproject to 3D camera coordinates (current frame)
                                        X_next = (u_next - self.cx) * depth_next / self.fx
                                        Y_next = (v_next - self.cy) * depth_next / self.fy
                                        Z_next = depth_next

                                        # Calculate 3D Euclidean distance (displacement in camera frame)
                                        displacement_3d = math.sqrt(
                                            (X_next - X_prev)**2 + 
                                            (Y_next - Y_prev)**2 + 
                                            (Z_next - Z_prev)**2
                                        )

                                        # Only consider reasonable displacements
                                        if displacement_3d > 0.01 and displacement_3d < 0.5:  # Sanity check
                                            displacements_3d.append(displacement_3d)

                            # Calculate median displacement to avoid outliers
                            if len(displacements_3d) > 0:
                                forward_distance = np.median(displacements_3d)
                                # Scale by 3 since we're processing every 3rd frame
                                forward_distance *= 0.2 
                                # Cap maximum displacement
                                forward_distance = min(forward_distance, 0.2)

                # Smooth translation
                self.smoothed_translation = (
                    self.motion_smooth_alpha * forward_distance +
                    (1 - self.motion_smooth_alpha) * self.smoothed_translation
                )

                # Update pose
                self.pose[2] += self.smoothed_yaw

                if self.is_rotating:
                    trans_step = 0.0
                else:
                    trans_step = self.smoothed_translation

                yaw_now = self.pose[2]
                self.pose[0] += trans_step * math.cos(yaw_now)
                self.pose[1] += trans_step * math.sin(yaw_now)

                # CRITICAL: Only update distance when NOT rotating
                if not self.is_rotating and trans_step > 1e-4:
                    old_distance = self.path_distance
                    self.path_distance += trans_step

                    # Check if we've crossed a meter boundary
                    old_meter = int(old_distance)
                    new_meter = int(self.path_distance)

                    if new_meter > old_meter:
                        # Store the position at this meter mark
                        self.meter_markers.append({
                            'x': self.pose[0],
                            'y': self.pose[1],
                            'distance': new_meter
                        })

                self.translation_magnitude = trans_step
                self.rotation_magnitude = abs(math.degrees(self.smoothed_yaw))

        self.trajectory.append(self.pose.copy())

        # ========== BUILD FEATURE MAP ==========
        # Only build map every 3rd frame (already processing every 3rd frame)
        self.current_features = self._detect_and_build_map(gray, depth_map, self.pose)

        # ========== UPDATE FEATURE TRACKING FOR DISTANCE CALCULATION ==========
        # Extract features for tracking (done every 3rd frame due to frame skipping)
        # Detect corners for tracking using Shi-Tomasi
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=50,  # Use 50 features for robust distance estimation
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )

        if corners is not None and len(corners) > 0:
            # Store feature positions as simple tuples for next processed frame
            # corners shape is (N, 1, 2), so we extract corners[:, 0, :]
            self.prev_tracked_features = [(float(pt[0][0]), float(pt[0][1])) for pt in corners]
        else:
            self.prev_tracked_features = None

        self.prev_gray = gray
        self.prev_depth = depth_map
        self.lookahead_depth_prev = self.lookahead_depth_curr

    def draw_camera_view(self, frame_bgr, depth_map):
        """Draw camera view with enhanced feature visualization"""
        # Create depth visualization
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

        # Blend with original frame
        vis = cv2.addWeighted(frame_bgr, 0.4, depth_colored, 0.6, 0)

        # Draw features with different colors based on detector type
        for item in self.current_features:
            if len(item) == 3:
                u, v, detector_type = item
                if detector_type == 'orb':
                    color = (0, 255, 255)  # Yellow for ORB
                elif detector_type == 'gftt':
                    color = (255, 255, 0)  # Cyan for GFTT
                else:  # fast
                    color = (255, 0, 255)  # Magenta for FAST
                cv2.circle(vis, (u, v), 2, color, -1)
            else:
                u, v = item
                cv2.circle(vis, (u, v), 2, (255, 255, 255), -1)

        # Draw lookahead point
        if self.lookahead_pixel is not None:
            u_look, v_look = self.lookahead_pixel
            cv2.circle(vis, (u_look, v_look), 10, (255, 255, 0), -1)
            cv2.circle(vis, (u_look, v_look), 14, (255, 255, 0), 2)
            cv2.line(vis, (u_look - 20, v_look), (u_look + 20, v_look), (255, 255, 0), 2)
            cv2.line(vis, (u_look, v_look - 20), (u_look, v_look + 20), (255, 255, 0), 2)

            if self.lookahead_depth_curr is not None:
                depth_text = f"{self.lookahead_depth_curr:.2f}m"
                cv2.putText(vis, depth_text, (u_look + 20, v_look - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Enhanced telemetry with detector breakdown
        y_offset = 25
        cv2.putText(vis, f"Features: ORB={self.feature_counts['orb']} "
                    f"GFTT={self.feature_counts['gftt']} "
                    f"FAST={self.feature_counts['fast']}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y_offset += 25
        cv2.putText(vis, f"Total: {self.feature_counts['total']} | Landmarks: {len(self.landmarks)}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y_offset += 25
        rot_color = (0, 255, 255) if self.is_rotating else (255, 200, 0)
        rot_text = f"Rotation: {self.rotation_magnitude:.2f}deg"
        if self.is_rotating:
            rot_text += " [ROTATING]"
        cv2.putText(vis, rot_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rot_color, 2)

        y_offset += 25
        trans_color = (100, 100, 100) if self.is_rotating else (0, 255, 0)
        trans_text = f"Forward: {self.translation_magnitude:.3f}m"
        if self.is_rotating:
            trans_text += " [SUPPRESSED]"
        cv2.putText(vis, trans_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, trans_color, 2)

        y_offset += 25
        cv2.putText(vis, "ROVER STREAM + ENHANCED DETECTION", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return vis

    def _project_world_to_top(self, Xw, Yw, cam_x, cam_y, scale, screen_cx, screen_cy):
        """Project world coordinates to screen coordinates"""
        dx = Xw - cam_x
        dy = Yw - cam_y
        sx = int(screen_cx + dx * scale)
        sy = int(screen_cy - dy * scale)
        return sx, sy

    def draw_top_view(self, scale=100, size=900, fps=None):
        """Draw top-down map"""
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        center_x, center_y = size // 2, size // 2

        cam_x, cam_y, cam_yaw = self.pose
        cam_screen_x = center_x
        cam_screen_y = int(size * 0.70)
        self._last_top_map_center = (cam_screen_x, cam_screen_y)

        # Draw Landmarks
        for lm in self.landmarks.values():
            sx, sy = self._project_world_to_top(lm['x'], lm['y'], cam_x, cam_y, scale, cam_screen_x, cam_screen_y)
            if 0 <= sx < size and 0 <= sy < size:
                brightness = int(np.clip(200.0 / (lm['z'] + 0.5), 40, 255))
                cv2.circle(canvas, (sx, sy), 2, (brightness, brightness, brightness), -1)

        # Draw Trajectory
        if len(self.trajectory) > 1:
            points = []
            for (x, y, _) in self.trajectory:
                px, py = self._project_world_to_top(x, y, cam_x, cam_y, scale, cam_screen_x, cam_screen_y)
                if 0 <= px < size and 0 <= py < size:
                    points.append((px, py))

            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(canvas, points[i], points[i + 1], (0, 180, 255), 2, cv2.LINE_AA)

        # Draw Meter Markers on the path
        for marker in self.meter_markers:
            mx, my = self._project_world_to_top(marker['x'], marker['y'], cam_x, cam_y, scale, cam_screen_x, cam_screen_y)
            if 0 <= mx < size and 0 <= my < size:
                # Draw a bright green circle for each meter
                cv2.circle(canvas, (mx, my), 6, (0, 255, 0), -1)
                cv2.circle(canvas, (mx, my), 8, (255, 255, 255), 2)

                # Draw the meter number
                meter_text = f"{marker['distance']}m"
                cv2.putText(canvas, meter_text, (mx + 12, my + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw Lookahead Point
        if self.lookahead_world_pos is not None:
            Xw, Yw, _ = self.lookahead_world_pos
            sx, sy = self._project_world_to_top(Xw, Yw, cam_x, cam_y, scale, cam_screen_x, cam_screen_y)
            if 0 <= sx < size and 0 <= sy < size:
                cv2.circle(canvas, (sx, sy), 8, (0, 255, 255), -1)
                cv2.circle(canvas, (sx, sy), 12, (0, 255, 255), 2)

        # Draw Camera Pose
        arrow_length = 40
        arrow_ex = int(cam_screen_x + arrow_length * math.cos(cam_yaw))
        arrow_ey = int(cam_screen_y - arrow_length * math.sin(cam_yaw))
        cv2.circle(canvas, (cam_screen_x, cam_screen_y), 8, (0, 255, 0), -1)

        arrow_color = (0, 255, 0) if not self.is_rotating else (0, 165, 255)
        cv2.arrowedLine(canvas, (cam_screen_x, cam_screen_y), (arrow_ex, arrow_ey),
                        arrow_color, 3, tipLength=0.3)

        if self.is_rotating:
            cv2.circle(canvas, (cam_screen_x, cam_screen_y), 35, (0, 165, 255), 2)

        # Overlay Information
        y_text = 30
        if fps is not None:
            cv2.putText(canvas, f"FPS: {fps:.1f}", (10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            y_text += 30

        cv2.putText(canvas, f"Distance: {self.path_distance:.2f}m", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_text += 30

        cv2.putText(canvas, f"Landmarks: {len(self.landmarks)}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y_text += 30

        cv2.putText(canvas, f"Features: {self.feature_counts['total']}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y_text += 30

        cv2.putText(canvas, f"Yaw: {math.degrees(cam_yaw):.1f}deg", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        y_text += 30

        status_text = "MODE: ROTATING" if self.is_rotating else "MODE: FORWARD"
        status_color = (0, 165, 255) if self.is_rotating else (0, 255, 0)
        cv2.putText(canvas, status_text, (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.putText(canvas, "ROVER STREAM | Fast: CLAHE+FAST",
                    (10, size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        return canvas

    def save_map_data(self, map_filename="map.pkl"):
        """Save map landmarks and features to pickle file"""
        try:
            with open(map_filename, "wb") as f:
                pickle.dump(self.landmarks, f)
            print(f"✅ Map data saved to: {map_filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving map: {e}")
            return False

    def run_slam(self, source=0, target_w=640, target_h=480, fx=None, fy=None, use_rover_stream=False, rover_url=None):
        """Main SLAM loop with rover stream support"""

        # Initialize capture based on source type
        if use_rover_stream and rover_url:
            cap = RoverStreamCapture(rover_url)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print("❌ Cannot open video source")
            return

        if fx is None or fy is None:
            fx = fy = 600.0

        print("\n" + "=" * 70)
        print("🎯 LOOKAHEAD SLAM v11 - ROVER STREAM + PREDICTION")
        print("=" * 70)
        print("✅ Fast feature detection: CLAHE + FAST only")
        print("✅ Arrow ONLY moves FORWARD or rotates in place")
        print("🚫 NO backward motion allowed")
        print("✅ Rotation only = Arrow rotates IN PLACE")
        print("✅ NO translation during rotation")
        print("✅ Distance accumulated only during forward translation")
        if use_rover_stream:
            print(f"📡 Using Rover IP Camera Stream: {rover_url}")
        print("📌 Cyan point (camera) = Ground lookahead pixel")
        print("🟡 Yellow dot (top map) = Lookahead world point")
        print("\n⌨️ Press ESC to quit")
        print("=" * 70 + "\n")

        fps = 0.0
        prev_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (target_w, target_h))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth = self.depth_map(frame)

            self.update(gray, depth)

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.85 * fps + 0.15 * (1.0 / dt) if fps > 0 else 1.0 / dt

            # Visualize
            vis = self.draw_camera_view(frame, depth)
            cv2.imshow("Rover Camera View", vis)

            top_map = self.draw_top_view(scale=100, size=900, fps=fps)
            cv2.imshow("Top-View Map (FAST Detection)", top_map)

            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

        # Save the final map
        print("\n💾 Saving final map...")
        final_map = self.draw_top_view(scale=100, size=900, fps=None)

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        map_filename = f"slam_map_{timestamp}.png"
        cv2.imwrite(map_filename, final_map)
        self.last_map_filename = map_filename
        print(f"✅ Map image saved as: {map_filename}")

        # Save map data (landmarks with features and depth)
        self.save_map_data("map.pkl")

        print(f"\n✅ SLAM completed:")
        print(f"   Total distance traveled: {self.path_distance:.2f} meters (forward only)")
        print(f"   Meter markers placed: {len(self.meter_markers)}")
        print(f"   Total landmarks created: {len(self.landmarks)}")
        print(f"   Trajectory points: {len(self.trajectory)}")
        print(f"   Feature detection: Fast (CLAHE + FAST only)")
        print(f"   Motion mode: FORWARD-ONLY")
        print(f"   Final map saved: {map_filename}")


# ========== PREDICTION MODULE ==========

def predict_image_pose_on_map(img_path, map_landmarks, slam_instance=None):
    """
    Predict the location of an image on the built map using feature matching
    Returns: (x_pred, y_pred, confidence)
    """
    # Load and preprocess image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Could not load image: {img_path}")
        return None, None, 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance image using same preprocessing as SLAM
    clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(3, 3))
    enhanced = clahe.apply(gray)
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.8, beta=15)

    # Extract ORB features
    orb = cv2.ORB_create(nfeatures=200)
    kp, desc = orb.detectAndCompute(enhanced, None)

    if desc is None or len(desc) == 0:
        print("❌ No features detected in the image")
        return None, None, 0.0

    # Gather all descriptors from map landmarks
    map_descs = []
    map_coords = []
    map_depths = []

    for lmk in map_landmarks.values():
        if lmk['descriptor'] is not None:
            map_descs.append(lmk['descriptor'])
            map_coords.append((lmk['x'], lmk['y']))
            map_depths.append(lmk['z'])

    if not map_descs:
        print("❌ No feature descriptors found in the map")
        return None, None, 0.0

    map_descs = np.stack(map_descs, axis=0)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc, map_descs)

    if len(matches) == 0:
        print("❌ No matches found between image and map")
        return None, None, 0.0

    # Sort matches by distance (lower is better)
    matches = sorted(matches, key=lambda m: m.distance)

    # Take best N matches
    n_best = min(10, len(matches))
    best_matches = matches[:n_best]

    # Extract matched coordinates
    xs = [map_coords[m.trainIdx][0] for m in best_matches]
    ys = [map_coords[m.trainIdx][1] for m in best_matches]
    depths = [map_depths[m.trainIdx] for m in best_matches]

    # Compute predicted position (mean of matched points)
    x_pred = np.mean(xs)
    y_pred = np.mean(ys)
    depth_pred = np.mean(depths)

    # Calculate confidence score (normalized by max ORB descriptor distance)
    # ORB descriptors use Hamming distance, max is typically 256 for 256-bit descriptors
    max_distance = 256.0
    avg_distance = np.mean([m.distance for m in best_matches])
    confidence = max(0.0, 1.0 - (avg_distance / max_distance))

    print(f"\n✅ Image localization successful!")
    print(f"   Matched features: {len(best_matches)}/{len(desc)}")
    print(f"   Average match distance: {avg_distance:.2f}")
    print(f"   Predicted position: ({x_pred:.2f}, {y_pred:.2f})")
    print(f"   Estimated depth: {depth_pred:.2f}m")
    print(f"   Confidence: {confidence*100:.1f}%")

    return x_pred, y_pred, confidence


def plot_prediction_on_map(x, y, confidence, map_image_path="slam_map_latest.png"):
    """
    Overlay the predicted position on the saved map image
    """
    # Find the most recent map file if default doesn't exist
    if not os.path.exists(map_image_path):
        # Find all slam_map files
        import glob
        map_files = glob.glob("slam_map_*.png")
        if map_files:
            map_files.sort(reverse=True)  # Get most recent
            map_image_path = map_files[0]
        else:
            print(f"❌ No map image found")
            return

    # Load map image
    map_img = cv2.imread(map_image_path)
    if map_img is None:
        print(f"❌ Could not load map image: {map_image_path}")
        return

    size = map_img.shape[0]
    scale = 100  # Same as used in draw_top_view

    # Convert world coordinates to screen coordinates
    # Assuming camera is at center-bottom (same as draw_top_view)
    cam_screen_x = size // 2
    cam_screen_y = int(size * 0.70)

    # For prediction, we assume the camera position is at origin (0, 0)
    # This is a simplification - ideally we'd use the last camera pose
    sx = int(cam_screen_x + x * scale)
    sy = int(cam_screen_y - y * scale)

    # Draw prediction marker
    if 0 <= sx < size and 0 <= sy < size:
        # Draw a red cross marker
        marker_size = 20
        cv2.drawMarker(map_img, (sx, sy), (0, 0, 255), 
                       markerType=cv2.MARKER_CROSS, markerSize=marker_size, thickness=3)

        # Draw a circle around it
        cv2.circle(map_img, (sx, sy), 15, (0, 0, 255), 2)

        # Add text with position and confidence
        text = f"Pred: ({x:.2f}, {y:.2f})"
        conf_text = f"Conf: {confidence*100:.1f}%"

        cv2.putText(map_img, text, (sx + 25, sy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(map_img, conf_text, (sx + 25, sy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add title
        cv2.putText(map_img, "IMAGE LOCATION PREDICTION", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Show the result
    cv2.imshow("Prediction on Map", map_img)
    print(f"\n🖼️ Displaying prediction on map. Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_filename = f"prediction_result_{timestamp}.png"
    cv2.imwrite(result_filename, map_img)
    print(f"✅ Prediction result saved as: {result_filename}")


# ========== MENU SYSTEM ==========

def main_menu():
    """Main menu for map building or prediction"""
    print("\n" + "=" * 70)
    print("🗺️  SLAM MAP BUILDER & PREDICTOR (with Rover Stream)")
    print("=" * 70)
    print("Choose an option:")
    print("1. Build Map (Rover Stream/Video)")
    print("2. Predict Image Location on Map")
    print("=" * 70)

    choice = input("Enter 1 or 2: ").strip()

    if choice == '1':
        run_slam_menu()
    elif choice == '2':
        predict_on_map_menu()
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")
        main_menu()


def run_slam_menu():
    """Menu for building the map with rover stream support"""
    print("\n" + "=" * 70)
    print("📹 MAP BUILDING MODE")
    print("=" * 70)
    print("Input source:")
    print("0 - Rover IP Camera Stream (http://10.47.11.127:8080/video_feed)")
    print("1 - Video file")
    print("=" * 70)

    mode = input("Enter 0 or 1: ").strip()

    # Create SLAM instance
    slam = LookaheadSLAM(
        fx=600.0,
        fy=600.0,
        cx=320.0,
        cy=240.0,
        lookahead_height_ratio=0.9
    )

    if mode == '0':
        # Use rover stream
        rover_url = "http://10.47.11.127:8080/video_feed"
        print(f"\n📡 Using Rover IP Camera Stream...")
        print(f"   URL: {rover_url}")
        slam.run_slam(source=None, use_rover_stream=True, rover_url=rover_url)
    elif mode == '1':
        video_path = input("Enter video file path: ").strip()
        if os.path.exists(video_path):
            print(f"\n🎥 Using video file: {video_path}")
            slam.run_slam(source=video_path, use_rover_stream=False)
        else:
            print(f"❌ Video file not found: {video_path}")
            return
    else:
        print("❌ Invalid choice.")
        return

    print("\n✅ Map building completed!")


def predict_on_map_menu():
    """Menu for predicting image location on map"""
    print("\n" + "=" * 70)
    print("🔍 IMAGE LOCATION PREDICTION MODE")
    print("=" * 70)

    # Check if map exists
    if not os.path.exists("map.pkl"):
        print("❌ No map found!")
        print("   Please build a map first using option 1.")
        print("=" * 70)

        choice = input("\nWould you like to build a map now? (y/n): ").strip().lower()
        if choice == 'y':
            run_slam_menu()
            # After building, ask again
            predict_on_map_menu()
        return

    print("✅ Map file found: map.pkl")
    print("=" * 70)

    # Get image path from user
    img_path = input("\nEnter path to image for prediction: ").strip()

    if not os.path.exists(img_path):
        print(f"❌ Image file not found: {img_path}")
        return

    # Load map data
    print("\n📂 Loading map data...")
    try:
        with open("map.pkl", "rb") as f:
            map_landmarks = pickle.load(f)
        print(f"✅ Map loaded successfully ({len(map_landmarks)} landmarks)")
    except Exception as e:
        print(f"❌ Error loading map: {e}")
        return

    # Perform prediction
    print("\n🔍 Analyzing image and matching features...")
    x, y, confidence = predict_image_pose_on_map(img_path, map_landmarks)

    if x is not None and y is not None:
        # Visualize prediction on map
        plot_prediction_on_map(x, y, confidence)
    else:
        print("❌ Prediction failed. Could not localize image on map.")


def run_slam(source=0, target_w=640, target_h=480, fx=None, fy=None):
    """Helper function to match original entry point signature"""
    slam = LookaheadSLAM(
        fx=fx if fx is not None else 600.0,
        fy=fy if fy is not None else 600.0,
        cx=target_w / 2.0,
        cy=target_h / 2.0,
        lookahead_height_ratio=0.9
    )

    slam.run_slam(source, target_w, target_h, fx, fy)


if __name__ == "__main__":
    main_menu()
