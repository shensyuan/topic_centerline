import cv2
import numpy as np
import os
from skimage.morphology import medial_axis
from scipy.ndimage import distance_transform_edt, label
from scipy.interpolate import splprep, splev

def is_valid_centerline(x_pts, y_pts, img_w=224):
    if len(x_pts) < 15: return False
    
    width = np.max(x_pts) - np.min(x_pts)
    y_variability = np.sum(np.abs(np.diff(y_pts))) / (width + 1e-6)
    
    # 1. 過濾寬度不足的雜訊
    if width < 20: return False 
    # 2. 過濾邊緣劇烈震盪
    if y_variability > 1.5: 
        return False

    return True

def batch_process_centerlines(input_folder):
    output_folder = input_folder.rstrip('\\/ ') + "_result_v4"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ('.png', '.jpg', '.jpeg')
    img_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions) and "_label" not in f]

    for img_name in img_files:
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(input_folder, f"{base_name}_label.png")
        img_path = os.path.join(input_folder, img_name)

        if not os.path.exists(label_path): continue

        img_orig = cv2.imread(img_path)
        mask_224 = cv2.imread(label_path, 0)
        if img_orig is None or mask_224 is None: continue
        
        h_orig, w_orig = img_orig.shape[:2]
        scale_x, scale_y = w_orig / 224.0, h_orig / 224.0

        _, binary_mask = cv2.threshold(mask_224, 127, 255, cv2.THRESH_BINARY)
        labeled_array, num_features = label(binary_mask > 0)
        
        result = img_orig.copy()
        
        for i in range(1, num_features + 1):
            single_region = (labeled_array == i)
            if np.sum(single_region) < 20: continue

            dist_map = distance_transform_edt(single_region)
            skel_region = medial_axis(single_region)
            y_idx, x_idx = np.where(skel_region)
            
            region_best_pts = {}
            for x, y in zip(x_idx, y_idx):
                val = dist_map[y, x]
                if x not in region_best_pts or val > region_best_pts[x][1]:
                    region_best_pts[x] = (y, val)
            
            sorted_x = np.array(sorted(region_best_pts.keys()))
            sorted_y = np.array([region_best_pts[x][0] for x in sorted_x])

            if not is_valid_centerline(sorted_x, sorted_y):
                continue

            pts_scaled = np.column_stack((sorted_x * scale_x, sorted_y * scale_y))

            try:
                k_val = min(3, len(pts_scaled)-1)
                # s 值決定平滑度，細線建議維持較高的平滑度以確保視覺質感
                tck, u = splprep([pts_scaled[:, 0], pts_scaled[:, 1]], s=len(pts_scaled)*12, k=k_val)
                u_fine = np.linspace(0, 1, 500) # 增加取樣點讓線條更細膩

                x_fine, y_fine = splev(u_fine, tck)
                
                final_pts = np.column_stack((x_fine, y_fine)).astype(np.int32)
                
                cv2.polylines(result, [final_pts], False, (0, 0, 255), 1, cv2.LINE_AA)
            except:
                continue

        cv2.imwrite(os.path.join(output_folder, f"{base_name}_centerline.png"), result)
        print(f"Done: {base_name}")

if __name__ == "__main__":
    batch_process_centerlines(r"./images")