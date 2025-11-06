# depth_slam_full_fb80_fix.py

# Monocular SLAM with MiDaS depth, LK + forward-back filtering on the good subset,
# robust yaw fusion (Essential + Homography + Center + Affine),
# voxel-hash landmark map, and realistic motion integration.
# FIXED: Translation disabled during rotation, stricter translation validation

import cv2
import torch
import math
import numpy as np
import time
from collections import deque

# --------------- MiDaS depth ---------------
print("Loading MiDaS...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()
print("MiDaS ready on", device)

class DepthPnPSLAM:
    def __init__(
        self,
        fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        scale_depth_min=0.5, scale_depth_max=15.0,
        pnp_reproj_thresh=2.5,  # Stricter PnP threshold
        ema_alpha_trans=0.85,  # More smoothing for translation
        ema_alpha_rot=0.45,   # mild smoothing for realistic yaw
        trans_scale=2,      # Reduced from 4.5 - stricter translation
        rot_scale=4.2,        # gentle yaw gain
        voxel_size=0.05,
        landmark_beta=0.25,
        max_features=500,      # cap ~80 features
        redetect_every=12,
        yaw_redetect_thresh=0.010,  # reseed on turns
        rot_insert_thresh=0.020,    # treat as rotation-heavy
        trans_eps=0.0001,     # Higher threshold - stricter translation
        yaw_clip_rad=0.30,          # ~17 deg per frame
        reproj_px_thresh=3.5,       # landmark validation gate
        rotation_trans_suppress_thresh=0.0005  # NEW: suppress translation if rotating
    ):
        # Intrinsics
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=float)
        # Depth scaling
        self.scale_depth_min = scale_depth_min
        self.scale_depth_max = scale_depth_max
        # PnP
        self.pnp_reproj_thresh = pnp_reproj_thresh
        # State
        self.pose = np.zeros(3, dtype=float) # [x, y, yaw]
        self.delta_trans_smooth = np.zeros(2)
        self.delta_yaw_smooth = 0.0
        self.ema_alpha_trans = ema_alpha_trans
        self.ema_alpha_rot = ema_alpha_rot
        self.trans_scale = trans_scale
        self.rot_scale = rot_scale
        self.traj = []
        # Previous frame buffers
        self.prev_gray = None
        self.prev_depth = None
        self.prev_pts = None
        self.frame_idx = 0
        # Map: voxel-hashed landmarks
        self.voxel_size = voxel_size
        self.landmark_beta = landmark_beta
        self.landmarks = {} # (ix,iy) -> {'x','y','z','n'}
        # Tracking params
        self.max_features = max_features
        self.redetect_every = redetect_every
        self.yaw_redetect_thresh = yaw_redetect_thresh
        self.rot_insert_thresh = rot_insert_thresh
        self.trans_eps = trans_eps
        self.yaw_clip_rad = yaw_clip_rad
        self.reproj_px_thresh = reproj_px_thresh
        self.rotation_trans_suppress_thresh = rotation_trans_suppress_thresh
        # For visualization
        self.curr_rot_deg =0.0
        self.curr_tracked_uv = np.empty((0,2), dtype=np.float32) # green
        self.curr_rot_new_uv = np.empty((0,2), dtype=np.float32) # cyan
        self.center_uv = None # magenta
        self.curr_yaw = 0.0
        # Translation history
        self.trans_hist = deque(maxlen=5)
        # Rotation history for suppression
        self.is_rotating = False

    # ----------- helpers -----------
    def _voxel_key(self, x, y):
        return (int(math.floor(x/self.voxel_size)), int(math.floor(y/self.voxel_size)))
    
    def _update_landmark(self, xw, yw, zw):
        k = self._voxel_key(xw, yw)
        if k in self.landmarks:
            lm = self.landmarks[k]
            b = self.landmark_beta
            lm['x'] = (1-b)*lm['x'] + b*xw
            lm['y'] = (1-b)*lm['y'] + b*yw
            lm['z'] = (1-b)*lm['z'] + b*zw
            lm['n'] += 1
        else:
            self.landmarks[k] = {'x': xw, 'y': yw, 'z': zw, 'n': 1}

    def _detect_features(self, gray):
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=7
        )
        if pts is None:
            return np.empty((0,1,2), dtype=np.float32)
        term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        cv2.cornerSubPix(gray, pts, (5,5), (-1,-1), term)
        return pts

    def _cap_pairs(self, prev_arr, next_arr):
        if len(prev_arr) <= self.max_features:
            return prev_arr, next_arr
        idx = np.linspace(0, len(prev_arr)-1, self.max_features).astype(int)
        return prev_arr[idx], next_arr[idx]

    # ----------- depth -----------
    def depth_map(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = transform(rgb).to(device)
        with torch.no_grad():
            pred = midas(inp)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()
        d = pred.cpu().numpy()
        return cv2.normalize(d, None, self.scale_depth_min, self.scale_depth_max, cv2.NORM_MINMAX)

    def backproject(self, u, v, z):
        X = (u - self.cx)/self.fx * z
        Y = (v - self.cy)/self.fy * z
        return np.array([X, Y, z], dtype=float)

    # ----------- transforms -----------
    def cam_to_world_xy(self, Xc, Zc, yaw, xw, yw):
        c, s = math.cos(yaw), math.sin(yaw)
        Xw = xw + (c*Zc - s*Xc)
        Yw = yw + (s*Zc + c*Xc)
        return Xw, Yw

    def world_to_cam_xz(self, Xw, Yw, yaw, xw, yw):
        dx, dy = Xw - xw, Yw - yw
        c, s = math.cos(yaw), math.sin(yaw)
        Xc = -s*dx + c*dy
        Zc = c*dx + s*dy
        return Xc, Zc

    # ----------- rotation from essential (recoverPose) -----------
    def estimate_yaw_recoverpose(self, prev_uv, next_uv):
        if prev_uv.shape[0] < 5:
            return None, 0.0
        E, _ = cv2.findEssentialMat(prev_uv, next_uv, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.5)
        if E is None:
            return None, 0.0
        _, R, _, mask2 = cv2.recoverPose(E, prev_uv, next_uv, self.K)
        yaw = math.atan2(R[1,0], R[0,0])
        inlier_ratio = float(mask2.sum())/len(mask2) if mask2 is not None and len(mask2) else 0.0
        return yaw, inlier_ratio

    # ----------- homography yaw fallback -----------
    def estimate_yaw_homography(self, prev_uv, next_uv):
        if prev_uv.shape[0] < 8:
            return None, 0.0
        H, mask = cv2.findHomography(prev_uv, next_uv, cv2.RANSAC, 3.0)
        if H is None:
            return None, 0.0
        R_approx = np.linalg.inv(self.K) @ H @ self.K
        U, _, Vt = np.linalg.svd(R_approx)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = -R
        yaw = math.atan2(R[1,0], R[0,0])
        inlier_ratio = (int(mask.sum())/len(mask)) if mask is not None and len(mask) else 0.0
        return yaw, inlier_ratio

    # ----------- main update -----------
    def update(self, gray, depth_curr):
        h, w = gray.shape
        self.frame_idx += 1
        self.curr_tracked_uv = np.empty((0,2), dtype=np.float32)
        self.curr_rot_new_uv = np.empty((0,2), dtype=np.float32)
        self.center_uv = None

        if self.prev_gray is None or self.prev_depth is None:
            self.prev_gray = gray
            self.prev_depth = depth_curr
            self.prev_pts = self._detect_features(gray)
            return

        # periodic reseed
        if (self.frame_idx % self.redetect_every == 0) or \
           (self.prev_pts.shape[0] < max(12, int(0.5*self.max_features))):
            self.prev_gray = gray
            self.prev_depth = depth_curr
            self.prev_pts = self._detect_features(gray)
            return

        # Track features forward
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=(21,21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Center reference motion
        center_pt = np.array([[[self.cx, self.cy]]], dtype=np.float32)
        nxt_c, st_c, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, center_pt, None,
            winSize=(21,21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if nxt is None or st is None:
            self.prev_gray = gray
            self.prev_depth = depth_curr
            self.prev_pts = self._detect_features(gray)
            return

        # Forward good subset mask
        st = st.reshape(-1)
        mask_fwd = (st == 1)
        if mask_fwd.sum() < 8:
            self.prev_gray = gray
            self.prev_depth = depth_curr
            self.prev_pts = self._detect_features(gray)
            return

        prev_good_pts_cv = self.prev_pts[mask_fwd] # Nx1x2
        nxt_good_pts_cv = nxt[mask_fwd]
        prev_good_arr = prev_good_pts_cv.reshape(-1,2) # Nx2
        next_good_arr = nxt_good_pts_cv.reshape(-1,2) # Nx2

        # Forward-backward consistency
        prev_back_cv, st_b, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, nxt_good_pts_cv, None,
            winSize=(21,21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        if prev_back_cv is not None and st_b is not None:
            st_b = st_b.reshape(-1)
            prev_back_arr = prev_back_cv.reshape(-1,2)
            fb_err = np.linalg.norm(prev_back_arr - prev_good_arr, axis=1)
            keep_good = (st_b == 1) & (fb_err < 1.0)
            prev_good_arr = prev_good_arr[keep_good]
            next_good_arr = next_good_arr[keep_good]
        if len(prev_good_arr) < 8:
            self.prev_gray = gray
            self.prev_depth = depth_curr
            self.prev_pts = self._detect_features(gray)
            return

        good_prev, good_next = self._cap_pairs(prev_good_arr, next_good_arr)
        self.curr_tracked_uv = good_next.copy()

        # ========== ROTATION ESTIMATION (KEEP EXACTLY AS IS) ==========
        # Center-ray yaw
        yaw_center = 0.0
        if nxt_c is not None and st_c is not None and st_c[0,0] == 1:
            u_center = float(nxt_c[0,0,0]); v_center = float(nxt_c[0,0,1])
            self.center_uv = (u_center, v_center)
            yaw_center = (u_center - self.cx) / self.fx

        # Affine yaw (2D rigid)
        M, _ = cv2.estimateAffinePartial2D(
            good_prev, good_next, method=cv2.RANSAC,
            ransacReprojThreshold=3.0, confidence=0.99, refineIters=20
        )
        yaw_aff = 0.0
        if M is not None:
            a11, a12 = M[0,0], M[0,1]
            yaw_aff = math.atan2(a12, a11)

        # Essential yaw + Homography yaw
        yaw_rp, ir = self.estimate_yaw_recoverpose(
            good_prev.astype(np.float32), good_next.astype(np.float32)
        )
        yaw_h, hr = self.estimate_yaw_homography(
            good_prev.astype(np.float32), good_next.astype(np.float32)
        )

        # Robust yaw fusion
        yaws, ws = [], []
        if yaw_rp is not None: yaws.append(yaw_rp); ws.append(0.45 * min(1.0, ir))
        if yaw_h is not None: yaws.append(yaw_h); ws.append(0.35 * min(1.0, hr))
        yaws.append(yaw_center); ws.append(0.15)
        yaws.append(yaw_aff); ws.append(0.05)
        ws = np.array(ws, dtype=float)
        ws /= (ws.sum() + 1e-9)
        yaw_raw = float(np.dot(ws, np.array(yaws, dtype=float)))
        yaw_raw = float(np.clip(yaw_raw, -self.yaw_clip_rad, self.yaw_clip_rad))
        self.curr_rot_deg = math.degrees(yaw_raw)
        rot_heavy = (abs(yaw_raw) > self.rot_insert_thresh) or \
                    (yaw_rp is not None and ir > 0.5) or \
                    (yaw_h is not None and hr > 0.5)
        # ========== END ROTATION ESTIMATION ==========

        # ========== TRANSLATION ESTIMATION WITH ROTATION SUPPRESSION ==========
        # Build 3D correspondences using PnP
        obj_pts, img_pts, uv_curr = [], [], []
        for p0, p1 in zip(good_prev, good_next):
            u0, v0 = int(round(p0[0])), int(round(p0[1]))
            if 0 <= u0 < w and 0 <= v0 < h:
                z0 = self.prev_depth[v0, u0]
                if np.isfinite(z0) and z0 > 1e-3:
                    obj_pts.append(self.backproject(u0, v0, z0))
                    img_pts.append([float(p1[0]), float(p1[1])])
                    uv_curr.append((int(round(p1[0])), int(round(p1[1]))))

        # PnP for translation
        tc = np.zeros(3, dtype=float)
        pnp_inlier_ratio = 0.0
        
        if len(obj_pts) >= 8:  # Stricter: require more points
            obj_pts = np.array(obj_pts, dtype=float)
            img_pts = np.array(img_pts, dtype=float)
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                obj_pts, img_pts, self.K, None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=self.pnp_reproj_thresh,  # Stricter threshold
                iterationsCount=150,  # More iterations for robustness
                confidence=0.995  # Higher confidence
            )
            if ok and inl is not None and len(inl) >= 8:  # Stricter: require more inliers
                pnp_inlier_ratio = len(inl) / len(obj_pts)
                # Only accept translation if inlier ratio is high enough
                if pnp_inlier_ratio >= 0.6:  # At least 60% inliers
                    R, _ = cv2.Rodrigues(rvec)
                    Rc = R.T
                    tc = -(R.T @ tvec.reshape(3))  # camera motion
        
        # ========== KEY FIX: SUPPRESS TRANSLATION DURING ROTATION ==========
        # Check if currently rotating
        self.is_rotating = abs(yaw_raw) > self.rotation_trans_suppress_thresh
        
        # If rotating, zero out translation
        if self.is_rotating:
            tc[:] = 0.0
            # print(f"[ROTATING] yaw={yaw_raw:.4f} - Translation suppressed")
        # ========== END TRANSLATION SUPPRESSION ==========
        
        # ========== END TRANSLATION ESTIMATION ==========

        prev_pose = self.pose.copy()
        yaw_prev = prev_pose[2]

        # World translation using previous yaw
        c0, s0 = math.cos(yaw_prev), math.sin(yaw_prev)
        dx_world = c0*tc[2] - s0*tc[0]
        dy_world = s0*tc[2] + c0*tc[0]
        trans_delta = np.array([dx_world, dy_world]) * self.trans_scale
        
        # Stricter translation threshold
        if np.linalg.norm(trans_delta) < self.trans_eps:
            trans_delta[:] = 0.0

        self.trans_hist.append(trans_delta)
        trans_med = np.median(np.stack(self.trans_hist, axis=0), axis=0)
        yaw_delta = yaw_raw * self.rot_scale
        
        # Smoothing
        self.delta_trans_smooth = self.ema_alpha_trans*trans_med + \
                                  (1-self.ema_alpha_trans)*self.delta_trans_smooth
        self.delta_yaw_smooth = self.ema_alpha_rot*yaw_delta + \
                                (1-self.ema_alpha_rot)*self.delta_yaw_smooth
        
        # Update pose
        self.pose[:2] = prev_pose[:2] + self.delta_trans_smooth
        self.pose[2] = prev_pose[2] + self.delta_yaw_smooth
        self.curr_yaw = self.pose[2]
        self.traj.append(self.pose.copy())

        if abs(yaw_delta) > self.yaw_redetect_thresh:
            self.prev_pts = self._detect_features(gray)

        # Landmark insertion with reprojection validation
        def valid_and_insert_from_cam(u, v, z, use_current_pose=True):
            if not np.isfinite(z) or z <= 1e-3:
                return
            if use_current_pose:
                yawI, xI, yI = self.pose[2], self.pose[0], self.pose[1]
            else:
                yawI, xI, yI = prev_pose[2], prev_pose[0], prev_pose[1]
            Xc, Yc, Zc = self.backproject(u, v, z)
            Xw, Yw = self.cam_to_world_xy(Xc, Zc, yawI, xI, yI)
            Xc2, Zc2 = self.world_to_cam_xz(Xw, Yw, yawI, xI, yI)
            if Zc2 <= 1e-6:
                return
            u2 = self.fx * Xc2 / Zc2 + self.cx
            if abs(u2 - u) > self.reproj_px_thresh:
                return
            self._update_landmark(Xw, Yw, z)

        H, W = depth_curr.shape

        if rot_heavy:
            self.curr_rot_new_uv = np.array(uv_curr, dtype=np.float32) if len(uv_curr) else np.empty((0,2), np.float32)
            for (u1, v1) in uv_curr:
                if 0 <= u1 < W and 0 <= v1 < H:
                    z1 = depth_curr[v1, u1]
                    valid_and_insert_from_cam(u1, v1, z1, use_current_pose=True)
        else:
            for (u0, v0) in [(int(round(p[0])), int(round(p[1]))) for p in good_prev]:
                if 0 <= u0 < w and 0 <= v0 < h:
                    z0 = self.prev_depth[v0, u0]
                    valid_and_insert_from_cam(u0, v0, z0, use_current_pose=False)
            for (u1, v1) in uv_curr:
                if 0 <= u1 < W and 0 <= v1 < H:
                    z1 = depth_curr[v1, u1]
                    valid_and_insert_from_cam(u1, v1, z1, use_current_pose=True)

        self.prev_gray = gray
        self.prev_depth = depth_curr
        self.prev_pts = self.curr_tracked_uv.reshape(-1,1,2)

    # ----------- visualization -----------
    def draw_top_view(self, scale=100, size=900, fps=None, arrow_centered=True):
        canvas = np.zeros((size, size, 3), np.uint8)
        cx, cy = size//2, size//2
        x_cam, y_cam, yaw = self.pose
        # Landmarks
        for lm in self.landmarks.values():
            sx = int(cx + (lm['x'] - x_cam)*scale)
            sy = int(cy - (lm['y'] - y_cam)*scale)
            if 0 <= sx < size and 0 <= sy < size:
                bright = int(np.clip(220.0/(lm['z'] + 0.3), 30, 255))
                canvas[sy, sx] = (bright, bright, bright)
        # Trajectory
        if len(self.traj) > 1:
            pts = [(int(cx + (x - x_cam)*scale), int(cy - (y - y_cam)*scale)) for (x,y,_) in self.traj[-1200:]]
            for a,b in zip(pts[:-1], pts[1:]):
                if 0 <= a[0] < size and 0 <= a[1] < size and 0 <= b[0] < size and 0 <= b[1] < size:
                    cv2.line(canvas, a, b, (0,140,255), 3)
        # Pose arrow
        px, py = cx, cy
        ex, ey = int(px + 40*math.cos(yaw)), int(py - 40*math.sin(yaw))
        cv2.arrowedLine(canvas, (px,py), (ex,ey), (255,255,0), 3, tipLength=0.35)
        
        # Display rotation status
        if self.is_rotating:
            cv2.putText(canvas, "ROTATING", (10,78), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
        
        if fps is not None:
            cv2.putText(canvas, f"FPS:{fps:.1f}", (10,26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (190,190,190), 2)
            cv2.putText(canvas, f"Landmarks:{len(self.landmarks)}", (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (190,190,190), 2)
        if fps is not None:
            cv2.putText(canvas, f"Yaw:{self.curr_rot_deg:.1f}°",
                        (10,78), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0,255,255) if self.is_rotating else (200,200,200), 2)
        cv2.imshow("Top-View Map", canvas)

def draw_depth_features(frame_bgr, depth, tracked_uv, rot_new_uv, center_uv, yaw):
    d8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    col = cv2.applyColorMap(d8, cv2.COLORMAP_MAGMA)
    vis = cv2.addWeighted(frame_bgr, 0.35, col, 0.65, 0)
    for (u,v) in rot_new_uv.reshape(-1,2):
        u, v = int(round(u)), int(round(v))
        if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
            cv2.circle(vis, (u,v), 4, (255,255,0), -1) # cyan rotation-added
    for (u,v) in tracked_uv.reshape(-1,2):
        u, v = int(round(u)), int(round(v))
        if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
            cv2.circle(vis, (u,v), 3, (0,255,0), -1) # green tracked
    if center_uv is not None:
        u, v = int(center_uv[0]), int(center_uv[1])
        if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
            cv2.circle(vis, (u,v), 5, (255,0,255), 2) # magenta center ref
    ax, ay = 60, 60
    ex, ey = int(ax + 40*math.cos(yaw)), int(ay - 40*math.sin(yaw))
    cv2.arrowedLine(vis, (ax, ay), (ex, ey), (255,255,255), 2, tipLength=0.35)
    cv2.rectangle(vis, (10,10), (120,80), (255,255,255), 1)
    return vis

# ----------------- Main loop -----------------
def run(source=0, target_w=640, target_h=480, fx=None, fy=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open video/source")
        return
    if fx is None or fy is None:
        # rough fx from FOV 60–65 deg for 640x480
        fx = fy = 600.0
    slam = DepthPnPSLAM(fx=fx, fy=fy, cx=target_w/2.0, cy=target_h/2.0)
    print("Press ESC to quit.")
    prev_time = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (target_w, target_h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth = slam.depth_map(frame)
        slam.update(gray, depth)
        now = time.time()
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.85*fps + 0.15*(1.0/dt) if fps > 0 else 1.0/dt
        vis = draw_depth_features(frame, depth, slam.curr_tracked_uv, slam.curr_rot_new_uv, slam.center_uv, slam.curr_yaw)
        cv2.putText(vis, f"Tracks:{slam.curr_tracked_uv.shape[0]} RotNew:{slam.curr_rot_new_uv.shape[0]} FPS:{fps:.1f}",
                    (10, target_h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.imshow("Depth + Features", vis)
        slam.draw_top_view(scale=100, size=900, fps=fps, arrow_centered=True)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    choice = input("0-Webcam 1-Video file : ").strip()
    if choice == '0':
        run(0)
    else:
        path = r"C:\Users\nandu\Documents\SEM 3-1\Medibot\deepsearch_slam\turn_clip.mp4"
        run(path)
