import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import math

# ==========================================================
# 1️⃣ Load Precomputed Scale Factor (From Calibration)
# ==========================================================
try:
    scale_factor = np.load("depth_scale_factor.npy").item()
    print(f"✅ Loaded precomputed scale factor: {scale_factor:.6f}")
    print(f"💡 Formula: distance = {scale_factor:.2f} / midas_value\n")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load scale factor: {e}\n"
                      "Run compute_scale_from_two_images.py first!") from e

# ==========================================================
# 2️⃣ Load Image
# ==========================================================
img_path = "image2.jpg"  # Your test image
frame = cv2.imread(img_path)
if frame is None:
    raise FileNotFoundError(f"❌ Image not found at path: {img_path}")

frame = cv2.resize(frame, (640, 480))
print(f"📸 Processing image: {img_path}")

# ==========================================================
# 3️⃣ Run MiDaS Depth Estimation
# ==========================================================
print("Running MiDaS depth estimation...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
transforms_midas = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms_midas.small_transform
device = torch.device("cpu")
midas.to(device)

img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
input_batch = transform(img_rgb).to(device)

with torch.no_grad():
    prediction = midas(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_map = prediction.cpu().numpy()
depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
print("✅ Depth estimation complete\n")

# ==========================================================
# 4️⃣ Apply Scale Factor (INVERSE DEPTH FORMULA)
# ==========================================================
# MiDaS outputs inverse depth: real_distance = scale / midas_value
# Avoid division by zero
epsilon = 1e-6
metric_depth_map = np.divide(
    scale_factor, 
    depth_map + epsilon,
    where=depth_map > epsilon,
    out=np.full_like(depth_map, np.inf)
)

print(f"📏 Converted to metric depth using inverse formula")
print(f"   distance = {scale_factor:.2f} / midas_value\n")

# ==========================================================
# 5️⃣ Optional: Geometric Correction (If Needed)
# ==========================================================
# UNCOMMENT ONLY IF YOU WANT HORIZONTAL GROUND DISTANCE
# camera_height_m = 1.5         # Your camera height above ground
# camera_tilt_deg = 25.0        # Your camera tilt angle (degrees down from horizontal)
# tilt_rad = math.radians(camera_tilt_deg)
# 
# # Method 1: Pythagorean (if camera points straight ahead)
# metric_depth_map = np.sqrt(np.maximum(0, metric_depth_map**2 - camera_height_m**2))
# 
# # Method 2: Cosine projection (if camera is tilted down)
# # metric_depth_map = metric_depth_map * np.cos(tilt_rad)
# 
# print(f"🌍 Applied geometric correction:")
# print(f"   Camera height: {camera_height_m}m | Tilt: {camera_tilt_deg}°\n")

# ==========================================================
# 6️⃣ Interactive Point Selection & Distance Display
# ==========================================================
clicked_point = []
display_frame = frame.copy()

def click_event(event, x, y, flags, param):
    global display_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point.clear()
        clicked_point.append((x, y))
        display_frame = frame.copy()
        cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)

cv2.namedWindow("Distance Measurement")
cv2.setMouseCallback("Distance Measurement", click_event)

print("="*60)
print("🖱️  INTERACTIVE DISTANCE MEASUREMENT")
print("="*60)
print("• Click on any object to measure its distance")
print("• Press 'q' or ESC to quit")
print("• Press 'c' to clear the current point")
print("• Press 's' to save the current depth map")
print("="*60 + "\n")

while True:
    vis_frame = display_frame.copy()
    
    # Display distance if point is clicked
    if clicked_point:
        x, y = clicked_point[0]
        distance = metric_depth_map[y, x]
        midas_val = depth_map[y, x]
        
        if np.isfinite(distance) and distance > 0:
            # Distance text
            cv2.putText(vis_frame, f"Distance: {distance:.2f} m", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            # MiDaS value (for debugging)
            cv2.putText(vis_frame, f"MiDaS: {midas_val:.1f}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(vis_frame, "Distance: Invalid", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    # Create colored depth visualization (closer = warmer)
    # Invert for visualization: 1/distance makes closer objects brighter
    depth_for_vis = np.clip(metric_depth_map, 0, 20)  # Clip to 20m max
    depth_inv = 1.0 / (depth_for_vis + 0.1)
    depth_normalized = (depth_inv - depth_inv.min()) / (depth_inv.max() - depth_inv.min() + 1e-8)
    depth_colored = (depth_normalized * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_JET)
    
    # Add legend to depth map
    cv2.putText(depth_colored, "Red=Close, Blue=Far", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("Distance Measurement", vis_frame)
    cv2.imshow("Depth Visualization", depth_colored)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # Quit
        break
    elif key == ord('c'):  # Clear
        clicked_point.clear()
        display_frame = frame.copy()
        print("✓ Point cleared")
    elif key == ord('s'):  # Save
        cv2.imwrite("depth_map_colored.png", depth_colored)
        np.save("depth_map_metric.npy", metric_depth_map)
        print("✓ Saved: depth_map_colored.png and depth_map_metric.npy")

cv2.destroyAllWindows()

# ==========================================================
# 7️⃣ Final Output
# ==========================================================
if clicked_point:
    x, y = clicked_point[0]
    distance = metric_depth_map[y, x]
    midas_val = depth_map[y, x]
    
    print("\n" + "="*60)
    print("🎯 FINAL MEASUREMENT RESULT")
    print("="*60)
    print(f"📍 Pixel coordinates:     ({x}, {y})")
    print(f"🔢 MiDaS depth value:     {midas_val:.2f}")
    print(f"📐 Scale factor:          {scale_factor:.2f}")
    
    if np.isfinite(distance) and distance > 0:
        print(f"📏 Estimated distance:    {distance:.2f} meters")
        print(f"\n✓ Calculation: {scale_factor:.2f} / {midas_val:.2f} = {distance:.2f} m")
    else:
        print(f"📏 Estimated distance:    Invalid")
    print("="*60)
else:
    print("\nℹ️  No point was selected during measurement")

print("\n✅ Program completed successfully!")