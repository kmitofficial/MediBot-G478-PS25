import cv2
import torch
import math
import numpy as np
import time
import os
import pickle
import requests
from collections import defaultdict, deque
from scipy.spatial import KDTree
from scipy.optimize import least_squares

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


# =============== Data Structures ===============

class MapPoint:
    """Represents a 3D map point with quality tracking"""
    _id_counter = 0
    
    def __init__(self, x, y, z, descriptor, first_keyframe_id):
        self.id = MapPoint._id_counter
        MapPoint._id_counter += 1
        
        self.x = x
        self.y = y
        self.z = z
        self.descriptor = descriptor
        
        # Quality tracking
        self.first_keyframe_id = first_keyframe_id
        self.observations = {first_keyframe_id: None}  # keyframe_id -> keypoint_index
        self.predicted_visible_count = 0
        self.actual_visible_count = 1
        self.is_validated = False
        self.keyframes_since_creation = 0
        
    def add_observation(self, keyframe_id, keypoint_idx):
        """Add an observation of this map point"""
        if keyframe_id not in self.observations:
            self.observations[keyframe_id] = keypoint_idx
            self.actual_visible_count += 1
    
    def increment_predicted_visible(self):
        """Increment predicted visible count"""
        self.predicted_visible_count += 1
    
    def passes_quality_test(self):
        """Check if map point passes the three-keyframe survival test"""
        if self.keyframes_since_creation < 3:
            return None  # Not ready for testing yet
        
        # Must be observed in at least 3 keyframes
        if len(self.observations) < 3:
            return False
        
        # Visibility ratio must exceed 25%
        if self.predicted_visible_count == 0:
            return False
        
        visibility_ratio = self.actual_visible_count / self.predicted_visible_count
        return visibility_ratio >= 0.25
    
    def get_position(self):
        """Return 3D position as numpy array"""
        return np.array([self.x, self.y, self.z])


class Keyframe:
    """Represents a keyframe with pose, features, and relationships"""
    _id_counter = 0
    
    def __init__(self, frame_id, pose, keypoints, descriptors, depth_map=None):
        self.id = Keyframe._id_counter
        Keyframe._id_counter += 1
        
        self.frame_id = frame_id
        self.pose = pose.copy()  # [x, y, yaw]
        self.keypoints = keypoints
        self.descriptors = descriptors
        
        # Store depth values at keypoint locations (not full depth map)
        self.depth_values = {}
        if depth_map is not None:
            for idx, kp in enumerate(keypoints):
                u, v = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                    self.depth_values[idx] = depth_map[v, u]
        
        # Map point observations
        self.mappoint_ids = {}  # keypoint_idx -> mappoint_id
        
        # Covisibility relationships
        self.covisible_keyframes = {}  # keyframe_id -> shared_mappoints_count
        
    def add_mappoint_observation(self, keypoint_idx, mappoint_id):
        """Associate a keypoint with a map point"""
        self.mappoint_ids[keypoint_idx] = mappoint_id
    
    def get_covisible_keyframes(self, min_shared=15):
        """Get keyframes with at least min_shared common observations"""
        return [kf_id for kf_id, count in self.covisible_keyframes.items() 
                if count >= min_shared]


class CovisibilityGraph:
    """Manages relationships between keyframes"""
    
    def __init__(self, min_shared_points=15):
        self.min_shared_points = min_shared_points
        self.graph = defaultdict(dict)  # keyframe_id -> {other_keyframe_id -> shared_count}
    
    def update_connections(self, keyframe, all_keyframes, map_points):
        """Update covisibility graph when adding a new keyframe"""
        kf_id = keyframe.id
        
        # Find all map points observed by this keyframe
        observed_mappoint_ids = set(keyframe.mappoint_ids.values())
        
        # Check covisibility with other keyframes
        for other_kf in all_keyframes.values():
            if other_kf.id == kf_id:
                continue
            
            other_observed = set(other_kf.mappoint_ids.values())
            shared_count = len(observed_mappoint_ids & other_observed)
            
            if shared_count >= self.min_shared_points:
                self.graph[kf_id][other_kf.id] = shared_count
                self.graph[other_kf.id][kf_id] = shared_count
                
                # Update keyframe objects
                keyframe.covisible_keyframes[other_kf.id] = shared_count
                other_kf.covisible_keyframes[kf_id] = shared_count
    
    def get_local_keyframes(self, keyframe_id, k1_size=10):
        """Get K1 (direct neighbors) and K2 (neighbors of neighbors)"""
        if keyframe_id not in self.graph:
            return [], []
        
        # K1: Top covisible keyframes
        covisible = self.graph[keyframe_id]
        k1 = sorted(covisible.keys(), key=lambda x: covisible[x], reverse=True)[:k1_size]
        
        # K2: Neighbors of K1 that aren't in K1
        k2 = set()
        for k1_id in k1:
            if k1_id in self.graph:
                k2.update(self.graph[k1_id].keys())
        k2 = k2 - set(k1) - {keyframe_id}
        
        return k1, list(k2)
    
    def remove_keyframe(self, keyframe_id):
        """Remove a keyframe from the graph"""
        if keyframe_id in self.graph:
            # Remove all edges involving this keyframe
            for neighbor_id in list(self.graph[keyframe_id].keys()):
                if neighbor_id in self.graph:
                    del self.graph[neighbor_id][keyframe_id]
            del self.graph[keyframe_id]


# =============== Modern SLAM System ===============

class ModernSLAM:
    """
    Modern Visual SLAM with keyframe-based architecture
    Features:
    - Frontend/Backend separation
    - Keyframe management and culling
    - Covisibility graph
    - Map point quality testing
    - Local optimization
    - Efficient spatial indexing (KDTree)
    """
    
    def __init__(self, fx=600.0, fy=600.0, cx=320.0, cy=240.0,
                 max_depth_threshold=4.5):
        
        # Camera intrinsics
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        
        # Keyframe management
        self.keyframes = {}  # keyframe_id -> Keyframe
        self.map_points = {}  # mappoint_id -> MapPoint
        self.current_keyframe = None
        self.last_keyframe_frame_id = -100
        
        # Covisibility graph
        self.covisibility_graph = CovisibilityGraph(min_shared_points=15)
        
        # Spatial indexing for fast nearest neighbor queries
        self.mappoint_kdtree = None
        self.mappoint_ids_in_kdtree = []
        
        # Tracking state
        self.current_pose = np.zeros(3, dtype=float)  # [x, y, yaw]
        self.velocity = np.zeros(3, dtype=float)  # For motion model
        self.trajectory = []
        
        # Frame tracking
        self.frame_count = 0
        self.prev_gray = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Feature extraction (optimized for keyframes)
        self.orb = cv2.ORB_create(
            nfeatures=1000,  # Extract more features for keyframes
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=10,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=7
        )
        
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Keyframe insertion thresholds
        self.min_frames_between_keyframes = 50
        self.keyframe_translation_threshold = 0.1  # meters
        self.keyframe_rotation_threshold = math.radians(10)  # 10 degrees
        self.keyframe_tracking_ratio_threshold = 0.90
        
        # Map point management
        self.max_depth_threshold = max_depth_threshold
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.target_fps = 25
        
        # Visualization
        self.display_features = []
        
        print("✅ Modern SLAM initialized with keyframe architecture")
    
    # ========== FRONTEND: Tracking ==========
    
    def track_frame(self, gray, depth_map=None):
        """
        Track current frame (frontend operation)
        Returns: success, needs_keyframe
        """
        self.frame_count += 1
        
        # Extract features for current frame
        keypoints, descriptors = self._extract_and_filter_features(gray)
        
        if descriptors is None or len(keypoints) < 20:
            return False, False
        
        # Initialize if first frame
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return True, True  # Need first keyframe
        
        # Track features using optical flow or matching
        if self.current_keyframe is not None:
            success = self._track_with_local_map(keypoints, descriptors, depth_map)
        else:
            # First frame - just initialize
            success = True
        
        # Check if we need a new keyframe
        needs_keyframe = self._check_keyframe_insertion_criteria(keypoints, descriptors)
        
        # Update for next frame
        self.prev_gray = gray.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return success, needs_keyframe
    
    def _track_with_local_map(self, keypoints, descriptors, depth_map):
        """Track current frame against local map"""
        if not self.map_points:
            return False
        
        # Get visible map points using KDTree
        visible_mappoints = self._get_visible_mappoints()
        
        if not visible_mappoints:
            return False
        
        # Match current frame features with visible map points
        matches = self._match_frame_with_mappoints(descriptors, visible_mappoints)
        
        if len(matches) < 20:
            return False
        
        # Estimate pose using PnP
        success = self._estimate_pose_pnp(keypoints, matches, visible_mappoints, depth_map)
        
        # Update visibility predictions for map points
        for mp in visible_mappoints:
            if self._is_mappoint_visible(mp, self.current_pose):
                mp.increment_predicted_visible()
        
        return success
    
    def _get_visible_mappoints(self, radius=3.0):
        """Get map points near current camera position using KDTree"""
        if self.mappoint_kdtree is None or not self.mappoint_ids_in_kdtree:
            return []
        
        # Query KDTree for nearby points
        cam_pos = self.current_pose[:2]  # [x, y]
        
        try:
            indices = self.mappoint_kdtree.query_ball_point(cam_pos, radius)
            visible_mp_ids = [self.mappoint_ids_in_kdtree[i] for i in indices]
            
            # Filter by viewing angle
            visible_mappoints = []
            for mp_id in visible_mp_ids:
                if mp_id in self.map_points:
                    mp = self.map_points[mp_id]
                    if self._is_mappoint_visible(mp, self.current_pose):
                        visible_mappoints.append(mp)
            
            return visible_mappoints
        except:
            return []
    
    def _is_mappoint_visible(self, mappoint, pose, max_distance=3.0, max_angle_deg=45):
        """Check if map point should be visible from given pose"""
        # Distance check
        cam_x, cam_y = pose[0], pose[1]
        distance = math.sqrt((mappoint.x - cam_x)**2 + (mappoint.y - cam_y)**2)
        
        if distance > max_distance or distance < 0.1:
            return False
        
        # Viewing angle check
        yaw = pose[2]
        cam_direction = np.array([math.cos(yaw), math.sin(yaw)])
        
        point_direction = np.array([mappoint.x - cam_x, mappoint.y - cam_y])
        point_direction_norm = np.linalg.norm(point_direction)
        
        if point_direction_norm < 1e-6:
            return False
        
        point_direction = point_direction / point_direction_norm
        
        angle = math.acos(np.clip(np.dot(cam_direction, point_direction), -1.0, 1.0))
        
        return angle < math.radians(max_angle_deg)
    
    def _match_frame_with_mappoints(self, frame_descriptors, mappoints):
        """Match frame features with map point descriptors"""
        if not mappoints or frame_descriptors is None:
            return []
        
        # Create descriptor matrix for map points
        mp_descriptors = np.array([mp.descriptor for mp in mappoints], dtype=np.uint8)
        
        # Match
        matches = self.bf_matcher.knnMatch(frame_descriptors, mp_descriptors, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) >= 2:
                m, n = match_pair[0], match_pair[1]
                if m.distance < 0.75 * n.distance:
                    good_matches.append((m.queryIdx, mappoints[m.trainIdx]))
        
        return good_matches
    
    def _estimate_pose_pnp(self, keypoints, matches, mappoints, depth_map):
        """Estimate camera pose using PnP with RANSAC"""
        if len(matches) < 10:
            return False
        
        # Prepare 2D-3D correspondences
        points_2d = []
        points_3d = []
        
        for kp_idx, mappoint in matches:
            if kp_idx < len(keypoints):
                kp = keypoints[kp_idx]
                points_2d.append(kp.pt)
                points_3d.append([mappoint.x, mappoint.y, mappoint.z])
        
        if len(points_2d) < 10:
            return False
        
        points_2d = np.array(points_2d, dtype=np.float32)
        points_3d = np.array(points_3d, dtype=np.float32)
        
        # Solve PnP with RANSAC
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d, points_2d, self.K, None,
                reprojectionError=8.0,
                confidence=0.99,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success and inliers is not None and len(inliers) >= 10:
                # Extract pose from solution
                # For 2D SLAM, we only use x, y, and yaw
                # This is a simplified extraction - in full 3D SLAM you'd use the full rotation
                self.current_pose[0] = tvec[0][0]
                self.current_pose[1] = tvec[2][0]  # Z in camera frame = Y in world
                
                # Extract yaw from rotation vector
                R, _ = cv2.Rodrigues(rvec)
                yaw = math.atan2(R[1, 0], R[0, 0])
                self.current_pose[2] = yaw
                
                return True
        except:
            pass
        
        return False
    
    def _check_keyframe_insertion_criteria(self, keypoints, descriptors):
        """Check if current frame should become a keyframe"""
        # Always insert first keyframe
        if self.current_keyframe is None:
            return True
        
        # Minimum frames since last keyframe
        frames_since_last_kf = self.frame_count - self.last_keyframe_frame_id
        if frames_since_last_kf < self.min_frames_between_keyframes:
            return False
        
        # Translation threshold
        translation = math.sqrt(
            (self.current_pose[0] - self.current_keyframe.pose[0])**2 +
            (self.current_pose[1] - self.current_keyframe.pose[1])**2
        )
        
        if translation > self.keyframe_translation_threshold:
            return True
        
        # Rotation threshold
        rotation = abs(self.current_pose[2] - self.current_keyframe.pose[2])
        rotation = (rotation + math.pi) % (2 * math.pi) - math.pi  # Normalize
        
        if abs(rotation) > self.keyframe_rotation_threshold:
            return True
        
        # Tracking quality (percentage of map points tracked)
        if self.current_keyframe and len(self.current_keyframe.mappoint_ids) > 0:
            # Try matching with current keyframe
            if self.prev_descriptors is not None and descriptors is not None:
                matches = self.bf_matcher.knnMatch(descriptors, self.prev_descriptors, k=2)
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) >= 2:
                        m, n = match_pair[0], match_pair[1]
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                
                tracking_ratio = len(good_matches) / len(self.current_keyframe.mappoint_ids)
                
                if tracking_ratio < self.keyframe_tracking_ratio_threshold:
                    return True
        
        return False
    
    # ========== BACKEND: Mapping ==========
    
    def insert_keyframe(self, gray, depth_map):
        """
        Insert a new keyframe (backend operation)
        This is where the heavy computation happens
        """
        # Extract high-quality features for keyframe
        keypoints, descriptors = self._extract_and_filter_features(gray, for_keyframe=True)
        
        if descriptors is None or len(keypoints) < 50:
            return False
        
        # Create keyframe
        keyframe = Keyframe(
            frame_id=self.frame_count,
            pose=self.current_pose.copy(),
            keypoints=keypoints,
            descriptors=descriptors,
            depth_map=depth_map
        )
        
        self.keyframes[keyframe.id] = keyframe
        self.current_keyframe = keyframe
        self.last_keyframe_frame_id = self.frame_count
        
        # Create new map points from this keyframe
        self._create_mappoints_from_keyframe(keyframe, gray, depth_map)
        
        # Update covisibility graph
        self.covisibility_graph.update_connections(keyframe, self.keyframes, self.map_points)
        
        # Local optimization (pose refinement)
        self._local_bundle_adjustment(keyframe)
        
        # Map point quality testing
        self._test_mappoint_quality(keyframe)
        
        # Keyframe culling (remove redundant keyframes)
        self._cull_redundant_keyframes(keyframe)
        
        # Rebuild KDTree for efficient spatial queries
        self._rebuild_kdtree()
        
        # Update trajectory
        self.trajectory.append(self.current_pose.copy())
        
        print(f"✅ KF#{keyframe.id}: {len(self.keyframes)} KFs, "
              f"{len(self.map_points)} MPs, Frame {self.frame_count}")
        
        return True
    
    def _create_mappoints_from_keyframe(self, keyframe, gray, depth_map):
        """Create new map points from keyframe observations"""
        created_count = 0
        
        for kp_idx, kp in enumerate(keyframe.keypoints):
            # Get depth at this keypoint
            if kp_idx not in keyframe.depth_values:
                continue
            
            midas_depth = keyframe.depth_values[kp_idx]
            metric_depth = self._midas_to_metric_depth(midas_depth)
            
            if metric_depth is None or metric_depth > self.max_depth_threshold:
                continue
            
            # Backproject to world coordinates
            world_pos = self._backproject_to_world(
                kp.pt[0], kp.pt[1], metric_depth, keyframe.pose
            )
            
            if world_pos is None:
                continue
            
            # Check if map point already exists at this location
            existing_mp = self._find_nearby_mappoint(world_pos[0], world_pos[1], threshold=0.05)
            
            if existing_mp is not None:
                # Associate with existing map point
                keyframe.add_mappoint_observation(kp_idx, existing_mp.id)
                existing_mp.add_observation(keyframe.id, kp_idx)
            else:
                # Create new map point
                descriptor = keyframe.descriptors[kp_idx]
                mappoint = MapPoint(
                    x=world_pos[0],
                    y=world_pos[1],
                    z=metric_depth,
                    descriptor=descriptor,
                    first_keyframe_id=keyframe.id
                )
                
                self.map_points[mappoint.id] = mappoint
                keyframe.add_mappoint_observation(kp_idx, mappoint.id)
                created_count += 1
        
        if created_count > 0:
            print(f"   Created {created_count} new map points")
    
    def _find_nearby_mappoint(self, x, y, threshold=0.05):
        """Find existing map point near given location"""
        for mp in self.map_points.values():
            distance = math.sqrt((mp.x - x)**2 + (mp.y - y)**2)
            if distance < threshold:
                return mp
        return None
    
    def _test_mappoint_quality(self, keyframe):
        """Test map point quality and remove low-quality points"""
        # Increment keyframes_since_creation for all map points
        for mp in self.map_points.values():
            mp.keyframes_since_creation += 1
        
        # Test map points that are ready
        to_remove = []
        for mp_id, mp in self.map_points.items():
            test_result = mp.passes_quality_test()
            
            if test_result is False:
                # Failed test - remove
                to_remove.append(mp_id)
            elif test_result is True:
                # Passed test - mark as validated
                mp.is_validated = True
        
        # Remove failed map points
        for mp_id in to_remove:
            self._remove_mappoint(mp_id)
        
        if to_remove:
            print(f"   Removed {len(to_remove)} low-quality map points")
    
    def _remove_mappoint(self, mp_id):
        """Remove a map point from the map"""
        if mp_id not in self.map_points:
            return
        
        mp = self.map_points[mp_id]
        
        # Remove from keyframe observations
        for kf_id in mp.observations.keys():
            if kf_id in self.keyframes:
                kf = self.keyframes[kf_id]
                # Remove from keyframe's mappoint_ids
                kf.mappoint_ids = {k: v for k, v in kf.mappoint_ids.items() if v != mp_id}
        
        # Remove from map
        del self.map_points[mp_id]
    
    def _cull_redundant_keyframes(self, new_keyframe):
        """
        Remove redundant keyframes using ORB-SLAM's "survival of the fittest" approach
        A keyframe is redundant if 90% of its map points are observed in at least 3 other keyframes
        """
        if len(self.keyframes) < 3:
            return
        
        to_remove = []
        
        for kf_id, kf in self.keyframes.items():
            # Don't cull the most recent keyframe
            if kf_id == new_keyframe.id:
                continue
            
            # Count map points and their redundancy
            observed_mappoints = list(kf.mappoint_ids.values())
            
            if not observed_mappoints:
                # No observations - can remove
                to_remove.append(kf_id)
                continue
            
            redundant_count = 0
            
            for mp_id in observed_mappoints:
                if mp_id not in self.map_points:
                    continue
                
                mp = self.map_points[mp_id]
                
                # Count how many OTHER keyframes observe this map point
                other_observations = sum(1 for obs_kf_id in mp.observations.keys() 
                                        if obs_kf_id != kf_id)
                
                if other_observations >= 3:
                    redundant_count += 1
            
            # Check redundancy ratio
            redundancy_ratio = redundant_count / len(observed_mappoints)
            
            if redundancy_ratio >= 0.90:
                to_remove.append(kf_id)
        
        # Remove redundant keyframes
        for kf_id in to_remove:
            self._remove_keyframe(kf_id)
        
        if to_remove:
            print(f"   Culled {len(to_remove)} redundant keyframes")
    
    def _remove_keyframe(self, kf_id):
        """Remove a keyframe from the map"""
        if kf_id not in self.keyframes:
            return
        
        kf = self.keyframes[kf_id]
        
        # Remove from covisibility graph
        self.covisibility_graph.remove_keyframe(kf_id)
        
        # Update map points (remove this keyframe from observations)
        for mp_id in kf.mappoint_ids.values():
            if mp_id in self.map_points:
                mp = self.map_points[mp_id]
                if kf_id in mp.observations:
                    del mp.observations[kf_id]
                
                # If map point has too few observations, remove it
                if len(mp.observations) < 2:
                    self._remove_mappoint(mp_id)
        
        # Remove from keyframes dict
        del self.keyframes[kf_id]
        
        # Update current_keyframe if needed
        if self.current_keyframe and self.current_keyframe.id == kf_id:
            if self.keyframes:
                # Set to most recent keyframe
                self.current_keyframe = max(self.keyframes.values(), 
                                           key=lambda k: k.frame_id)
    
    def _local_bundle_adjustment(self, keyframe):
        """
        Simplified local bundle adjustment (pose-only optimization)
        Optimizes the keyframe pose to minimize reprojection error
        """
        # Get map points observed in this keyframe
        observed_mappoints = []
        keypoint_indices = []
        
        for kp_idx, mp_id in keyframe.mappoint_ids.items():
            if mp_id in self.map_points:
                observed_mappoints.append(self.map_points[mp_id])
                keypoint_indices.append(kp_idx)
        
        if len(observed_mappoints) < 10:
            return
        
        # Optimize pose to minimize reprojection error
        initial_pose = keyframe.pose.copy()
        
        def residual_function(pose_params):
            """Compute reprojection errors"""
            pose = pose_params
            errors = []
            
            for mp, kp_idx in zip(observed_mappoints, keypoint_indices):
                # Project map point to image
                projected = self._project_to_image(mp.x, mp.y, mp.z, pose)
                
                if projected is None:
                    continue
                
                # Observed keypoint
                observed = keyframe.keypoints[kp_idx].pt
                
                # Reprojection error
                error = [projected[0] - observed[0], projected[1] - observed[1]]
                errors.extend(error)
            
            return np.array(errors)
        
        try:
            # Run optimization
            result = least_squares(
                residual_function,
                initial_pose,
                method='lm',
                max_nfev=20
            )
            
            # Update keyframe pose
            keyframe.pose = result.x
            self.current_pose = result.x.copy()
            
        except Exception as e:
            # Keep original pose on failure
            pass
    
    def _rebuild_kdtree(self):
        """Rebuild KDTree for efficient spatial queries of map points"""
        if not self.map_points:
            self.mappoint_kdtree = None
            self.mappoint_ids_in_kdtree = []
            return
        
        # Extract 2D positions
        positions = []
        mp_ids = []
        
        for mp_id, mp in self.map_points.items():
            positions.append([mp.x, mp.y])
            mp_ids.append(mp_id)
        
        self.mappoint_ids_in_kdtree = mp_ids
        self.mappoint_kdtree = KDTree(positions)
    
    # ========== Helper Functions ==========
    
    def _extract_and_filter_features(self, gray, for_keyframe=False):
        """Extract and filter ORB features with quality-based selection"""
        # Enhance image
        enhanced = self.clahe.apply(gray)
        
        # Detect features
        keypoints = self.orb.detect(enhanced, None)
        
        if not keypoints:
            return [], None
        
        # Sort by response (quality) and keep top features
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)
        
        if for_keyframe:
            # Keep more features for keyframes
            max_features = 500
        else:
            # Fewer features for tracking frames
            max_features = 200
        
        keypoints = keypoints[:max_features]
        
        # Apply spatial distribution filter
        keypoints = self._spatial_distribution_filter(keypoints, gray.shape, grid_size=8)
        
        # Compute descriptors
        keypoints, descriptors = self.orb.compute(enhanced, keypoints)
        
        return keypoints, descriptors
    
    def _spatial_distribution_filter(self, keypoints, image_shape, grid_size=8):
        """Ensure features are spatially distributed"""
        height, width = image_shape
        cell_h = height // grid_size
        cell_w = width // grid_size
        
        grid = defaultdict(list)
        
        # Assign keypoints to grid cells
        for kp in keypoints:
            cell_x = int(kp.pt[0] // cell_w)
            cell_y = int(kp.pt[1] // cell_h)
            grid[(cell_x, cell_y)].append(kp)
        
        # Keep best keypoint(s) from each cell
        filtered_keypoints = []
        max_per_cell = 3
        
        for cell_kps in grid.values():
            cell_kps_sorted = sorted(cell_kps, key=lambda x: x.response, reverse=True)
            filtered_keypoints.extend(cell_kps_sorted[:max_per_cell])
        
        return filtered_keypoints
    
    def _midas_to_metric_depth(self, midas_value):
        """Convert MiDaS disparity to metric depth"""
        if midas_value < 1e-3:
            return None
        return DEPTH_SCALE / (midas_value + 1e-6)
    
    def _backproject_to_world(self, u, v, metric_depth, pose):
        """Backproject pixel to world coordinates"""
        if metric_depth is None or metric_depth <= 0:
            return None
        
        # Camera frame
        X_cam = (u - self.cx) * metric_depth / self.fx
        Y_cam = (v - self.cy) * metric_depth / self.fy
        Z_cam = metric_depth
        
        # Transform to world frame
        yaw = pose[2]
        c, s = math.cos(yaw), math.sin(yaw)
        
        x_world = pose[0] + (c * Z_cam + s * X_cam)
        y_world = pose[1] + (s * Z_cam - c * X_cam)
        
        return np.array([x_world, y_world, metric_depth])
    
    def _project_to_image(self, x, y, z, pose):
        """Project world point to image coordinates"""
        # World to camera frame
        yaw = pose[2]
        c, s = math.cos(yaw), math.sin(yaw)
        
        dx = x - pose[0]
        dy = y - pose[1]
        
        X_cam = c * dx + s * dy
        Z_cam = -s * dx + c * dy
        
        if Z_cam <= 0:
            return None
        
        # Camera to image
        u = self.fx * (X_cam / Z_cam) + self.cx
        v = self.fy * (0 / Z_cam) + self.cy  # Simplified for ground plane
        
        return (u, v)
    
    # ========== Main SLAM Loop ==========
    
    def run_slam(self, source=0, use_rover_stream=False, rover_url=None):
        """Main SLAM loop with frontend/backend separation"""
        # Open video source
        if use_rover_stream and rover_url:
            cap = RoverStreamCapture(rover_url)
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("❌ Failed to open video source")
            return
        
        print("🚀 Starting Modern SLAM...")
        print("   Press 'q' to quit")
        print("   Press 's' to save map")
        
        fps_timer = time.time()
        frame_times = deque(maxlen=30)
        
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize and convert to grayscale
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # FRONTEND: Track current frame
            success, needs_keyframe = self.track_frame(gray)
            
            # Compute depth only for keyframes (huge optimization)
            depth_map = None
            if needs_keyframe:
                depth_map = self._compute_depth_map(frame)
            
            # BACKEND: Insert keyframe if needed
            if needs_keyframe and depth_map is not None:
                self.insert_keyframe(gray, depth_map)
            
            # Visualization
            vis_frame = self._visualize_tracking(frame, gray)
            
            # Calculate FPS
            loop_time = time.time() - loop_start
            frame_times.append(loop_time)
            fps = 1.0 / (sum(frame_times) / len(frame_times))
            self.fps_history.append(fps)
            
            # Display info
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Frame: {self.frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Keyframes: {len(self.keyframes)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Map Points: {len(self.map_points)}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Pose: ({self.current_pose[0]:.2f}, {self.current_pose[1]:.2f})", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Modern SLAM", vis_frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_map()
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final save
        self.save_map()
        
        print("\n✅ SLAM finished")
        print(f"   Total frames: {self.frame_count}")
        print(f"   Total keyframes: {len(self.keyframes)}")
        print(f"   Total map points: {len(self.map_points)}")
        print(f"   Average FPS: {np.mean(list(self.fps_history)):.1f}")
    
    def _compute_depth_map(self, frame):
        """Compute MiDaS depth map (only for keyframes)"""
        try:
            # Prepare input
            input_batch = transform(frame).to(device)
            
            # Inference
            with torch.no_grad():
                prediction = midas(input_batch)
            
            # Resize to match frame
            depth = prediction.squeeze().cpu().numpy()
            depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
            
            return depth
        except Exception as e:
            print(f"⚠️  Depth computation failed: {e}")
            return None
    
    def _visualize_tracking(self, frame, gray):
        """Visualize current tracking state"""
        vis = frame.copy()
        
        # Draw tracked features if available
        if self.prev_keypoints is not None:
            for kp in self.prev_keypoints[:100]:  # Show subset for speed
                pt = (int(kp.pt[0]), int(kp.pt[1]))
                cv2.circle(vis, pt, 2, (0, 255, 0), -1)
        
        return vis
    
    def save_map(self):
        """Save the current map to disk"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"map_modern_{timestamp}.pkl"
        
        # Prepare map data
        map_data = {
            'keyframes': {},
            'map_points': {},
            'trajectory': self.trajectory,
            'covisibility_graph': dict(self.covisibility_graph.graph),
            'metadata': {
                'total_frames': self.frame_count,
                'num_keyframes': len(self.keyframes),
                'num_mappoints': len(self.map_points),
                'camera_matrix': self.K
            }
        }
        
        # Serialize keyframes
        for kf_id, kf in self.keyframes.items():
            map_data['keyframes'][kf_id] = {
                'id': kf.id,
                'frame_id': kf.frame_id,
                'pose': kf.pose,
                'keypoints': [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response) 
                             for kp in kf.keypoints],
                'descriptors': kf.descriptors,
                'depth_values': kf.depth_values,
                'mappoint_ids': kf.mappoint_ids
            }
        
        # Serialize map points
        for mp_id, mp in self.map_points.items():
            map_data['map_points'][mp_id] = {
                'id': mp.id,
                'x': mp.x,
                'y': mp.y,
                'z': mp.z,
                'descriptor': mp.descriptor,
                'observations': mp.observations,
                'is_validated': mp.is_validated
            }
        
        # Save to file
        with open(filename, 'wb') as f:
            pickle.dump(map_data, f)
        
        print(f"\n💾 Map saved: {filename}")
        print(f"   Keyframes: {len(self.keyframes)}")
        print(f"   Map points: {len(self.map_points)}")
        
        return filename


# ========== Localization (Testing) ==========

def load_map(filename):
    """Load a saved map"""
    with open(filename, 'rb') as f:
        map_data = pickle.load(f)
    
    print(f"\n📂 Loaded map: {filename}")
    print(f"   Keyframes: {len(map_data['keyframes'])}")
    print(f"   Map points: {len(map_data['map_points'])}")
    
    return map_data


def estimate_pose_modern(test_image_path, map_file):
    """Estimate pose using the modern SLAM map"""
    # Load map
    map_data = load_map(map_file)
    
    # Load test image
    test_img = cv2.imread(test_image_path)
    if test_img is None:
        print(f"❌ Could not load image: {test_image_path}")
        return
    
    test_img = cv2.resize(test_img, (640, 480))
    gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
    # Extract features from test image
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    kp, desc = orb.detectAndCompute(enhanced, None)
    
    print(f"\n🔍 Test image features: {len(kp)}")
    
    # Match with map points
    mp_descriptors = []
    mp_positions = []
    mp_ids = []
    
    for mp_id, mp_data in map_data['map_points'].items():
        mp_descriptors.append(mp_data['descriptor'])
        mp_positions.append([mp_data['x'], mp_data['y'], mp_data['z']])
        mp_ids.append(mp_id)
    
    mp_descriptors = np.array(mp_descriptors, dtype=np.uint8)
    
    # Match
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc, mp_descriptors, k=2)
    
    # Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) >= 2:
            m, n = match_pair[0], match_pair[1]
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    print(f"✅ Good matches: {len(good_matches)}")
    
    if len(good_matches) < 20:
        print("❌ Not enough matches for reliable pose estimation")
        return
    
    # Estimate position from matched features (simplified - using centroid)
    matched_positions = []
    for m in good_matches:
        mp_pos = mp_positions[m.trainIdx]
        matched_positions.append([mp_pos[0], mp_pos[1]])
    
    matched_positions = np.array(matched_positions)
    estimated_x = np.median(matched_positions[:, 0])
    estimated_y = np.median(matched_positions[:, 1])
    
    # Find nearest trajectory point
    trajectory = map_data['trajectory']
    min_dist = float('inf')
    best_seg_idx = 0
    
    for i in range(len(trajectory) - 1):
        p1 = np.array(trajectory[i][:2])
        p2 = np.array(trajectory[i+1][:2])
        
        # Distance to line segment
        est_pos = np.array([estimated_x, estimated_y])
        d = np.linalg.norm(np.cross(p2-p1, p1-est_pos)) / np.linalg.norm(p2-p1)
        
        if d < min_dist:
            min_dist = d
            best_seg_idx = i
    
    # Project onto trajectory
    p1 = np.array(trajectory[best_seg_idx][:2])
    p2 = np.array(trajectory[best_seg_idx+1][:2])
    est_pos = np.array([estimated_x, estimated_y])
    
    t = np.dot(est_pos - p1, p2 - p1) / np.dot(p2 - p1, p2 - p1)
    t = np.clip(t, 0, 1)
    
    proj_pos = p1 + t * (p2 - p1)
    x_final = proj_pos[0]
    y_final = proj_pos[1]
    
    # Estimate yaw
    yaw = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    
    print(f"\n✅ Estimated pose:")
    print(f"   Position: ({x_final:.3f}, {y_final:.3f})")
    print(f"   Yaw: {math.degrees(yaw):.1f}°")
    print(f"   Segment: {best_seg_idx}/{len(trajectory)-1}")
    print(f"   Distance to trajectory: {min_dist:.3f}m")
    
    # Visualize
    visualize_pose_modern(x_final, y_final, yaw, map_data, test_img, kp, good_matches)


def visualize_pose_modern(x, y, yaw, map_data, test_image, kp, matches):
    """Visualize estimated pose on map"""
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
    
    # Draw map points
    for mp_data in map_data['map_points'].values():
        sx, sy = project(mp_data['x'], mp_data['y'])
        if 0 <= sx < size and 0 <= sy < size:
            brightness = int(np.clip(200.0 / (mp_data['z'] + 0.5), 40, 255))
            cv2.circle(map_canvas, (sx, sy), 1, (brightness, brightness, brightness), -1)
    
    # Draw trajectory
    trajectory = map_data['trajectory']
    if len(trajectory) > 1:
        traj_points = []
        for pose in trajectory:
            if len(pose) >= 2:
                tx, ty = pose[0], pose[1]
                px, py = project(tx, ty)
                if 0 <= px < size and 0 <= py < size:
                    traj_points.append((px, py))
        
        for i in range(len(traj_points) - 1):
            cv2.line(map_canvas, traj_points[i], traj_points[i+1], 
                    (0, 165, 255), 5, cv2.LINE_AA)
    
    # Draw matched features
    mp_ids = list(map_data['map_points'].keys())
    for m in matches[:100]:  # Limit for visualization
        if m.trainIdx < len(mp_ids):
            mp_data = map_data['map_points'][mp_ids[m.trainIdx]]
            sx, sy = project(mp_data['x'], mp_data['y'])
            if 0 <= sx < size and 0 <= sy < size:
                cv2.circle(map_canvas, (sx, sy), 4, (0, 255, 0), -1)
    
    # Draw estimated pose
    PINK = (203, 192, 255)
    arrow_len = 50
    arrow_ex = int(cam_screen_x + arrow_len * math.cos(yaw))
    arrow_ey = int(cam_screen_y - arrow_len * math.sin(yaw))
    
    cv2.circle(map_canvas, (cam_screen_x, cam_screen_y), 18, PINK, -1)
    cv2.circle(map_canvas, (cam_screen_x, cam_screen_y), 24, PINK, 4)
    cv2.arrowedLine(map_canvas, (cam_screen_x, cam_screen_y), (arrow_ex, arrow_ey),
                   PINK, 5, tipLength=0.3)
    
    # Info overlay
    cv2.putText(map_canvas, "MODERN SLAM", (10, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, PINK, 3)
    cv2.putText(map_canvas, f"Pos: ({x:.2f}, {y:.2f})", (10, 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(map_canvas, f"Yaw: {math.degrees(yaw):.1f}deg", (10, 105),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(map_canvas, f"Matches: {len(matches)}", (10, 135),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Test image with matches
    test_vis = test_image.copy()
    for m in matches[:100]:
        pt = kp[m.queryIdx].pt
        cv2.circle(test_vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
    
    cv2.putText(test_vis, "TEST IMAGE", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, PINK, 2)
    
    cv2.imshow("Test Image", test_vis)
    cv2.imshow("Modern SLAM - Localization", map_canvas)
    
    print("\n📍 Visualization ready - Press any key...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"pose_modern_{timestamp}.png", map_canvas)
    print(f"✅ Saved: pose_modern_{timestamp}.png")


# ========== Menu System ==========

def main_menu():
    print("\n" + "="*70)
    print("🚀 MODERN SLAM - Keyframe-Based Architecture")
    print("="*70)
    print("1. Build Map (Modern SLAM)")
    print("2. Test Pose (Localization)")
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
    
    slam = ModernSLAM(
        fx=600.0, fy=600.0, cx=320.0, cy=240.0,
        max_depth_threshold=3.0
    )
    
    if choice == '0':
        slam.run_slam(source=None, use_rover_stream=True, 
                     rover_url="http://10.47.11.127:8080/video_feed")
    elif choice == '1':
        slam.run_slam(source=0)
    elif choice == '2':
        path = input("Video file path: ").strip()
        if os.path.exists(path):
            slam.run_slam(source=path)


def estimate_menu():
    print("\n🔍 TEST POSE")
    
    import glob
    maps = sorted(glob.glob("map_modern_*.pkl"), reverse=True)
    
    if not maps:
        print("❌ No maps found!")
        return
    
    print(f"✅ Using: {maps[0]}")
    
    img = input("📸 Test image path: ").strip()
    
    if os.path.exists(img):
        estimate_pose_modern(img, maps[0])
    else:
        print("❌ Image not found")


if __name__ == "__main__":
    main_menu()