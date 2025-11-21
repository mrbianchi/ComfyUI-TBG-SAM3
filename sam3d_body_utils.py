import os
import sys
import torch
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, Any

# Add sam-3d-body to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sam3d_body_path = os.path.join(parent_dir, "sam-3d-body")

if sam3d_body_path not in sys.path:
    sys.path.append(sam3d_body_path)

try:
    from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator
    from tools.vis_utils import visualize_sample_together
except ImportError as e:
    print(f"[SAM3D] Warning: Could not import sam_3d_body. Make sure the repo is cloned next to this node. Error: {e}")
    SAM3DBodyEstimator = None

class SAM3DBodyWrapper:
    def __init__(self, device="cuda"):
        self.device = device
        self.estimator = None
        self.model = None
        self.model_cfg = None

    def load_model(self, model_type="dinov3"):
        if SAM3DBodyEstimator is None:
            raise ImportError("SAM 3D Body modules not found. Please check installation.")

        model_id = f"facebook/sam-3d-body-{model_type}"
        print(f"[SAM3D] Loading model: {model_id}")
        
        try:
            # We need to find where the assets are for MHR
            # Assuming they are in the sam-3d-body repo under assets
            mhr_path = os.path.join(sam3d_body_path, "assets", "mhr_model.pt")
            if not os.path.exists(mhr_path):
                 # Try to find it in checkpoints if user followed demo instructions
                 mhr_path = os.path.join(sam3d_body_path, "checkpoints", "sam-3d-body-dinov3", "assets", "mhr_model.pt")
            
            if not os.path.exists(mhr_path):
                print(f"[SAM3D] Warning: MHR model not found at {mhr_path}. Some features might fail.")

            self.model, self.model_cfg = load_sam_3d_body_hf(model_id, device=self.device)
            
            self.estimator = SAM3DBodyEstimator(
                sam_3d_body_model=self.model,
                model_cfg=self.model_cfg,
            )
            print("[SAM3D] Model loaded successfully")
            return self
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM 3D Body model: {str(e)}")

    def process_image(self, image: np.ndarray, bbox_thr: float = 0.5, use_mask: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single image and return visualization and mask
        image: numpy array (H, W, 3) RGB
        """
        if self.estimator is None:
            raise RuntimeError("Model not loaded")

        # Save temp image because process_one_image expects a path
        # TODO: Modify SAM3DBody to accept numpy array directly if possible, 
        # but for now we'll use a temp file to be safe with their API
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            # Convert RGB to BGR for OpenCV
            cv2.imwrite(tmp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        try:
            outputs = self.estimator.process_one_image(
                tmp_path,
                bbox_thr=bbox_thr,
                use_mask=use_mask,
            )
            
            # Visualization
            # visualize_sample_together expects BGR image
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            vis_img = visualize_sample_together(img_bgr, outputs, self.estimator.faces)
            # Convert back to RGB
            vis_img_rgb = cv2.cvtColor(vis_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            
            # Create a mask from the projected vertices or segmentation
            # The outputs object likely contains masks or we can project vertices
            # For now, let's try to get a mask from the visualization or outputs
            # outputs is a list of dicts? Let's check demo.py usage
            # It seems outputs is what visualize_sample_together uses.
            
            # Let's create a simple mask of where the mesh is
            # This is a placeholder - ideally we render the mesh to a mask
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)
            
            # If we have 2d keypoints or masks in outputs, use them
            # Inspecting demo.py doesn't show structure of outputs, but it's likely a list of instances
            
            return vis_img_rgb, mask

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
