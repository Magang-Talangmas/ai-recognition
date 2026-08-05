"""
OpenVINO FP16 MTCNN Face Detector & Landmark Alignment
Optimized for high-FPS, ultra low-latency CPU execution using OpenVINO runtime.
"""

import os
import cv2
import numpy as np
import openvino as ov


def nms(boxes, threshold, method="Union"):
    """Fast Non-Maximum Suppression"""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int32)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        if method == "Min":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
        
    return np.array(keep, dtype=np.int32)


def calibrate_boxes(boxes, regressions):
    """Calibrate bounding boxes using regression offsets"""
    if len(boxes) == 0:
        return boxes
    
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    
    boxes[:, 0] = boxes[:, 0] + regressions[:, 0] * w
    boxes[:, 1] = boxes[:, 1] + regressions[:, 1] * h
    boxes[:, 2] = boxes[:, 2] + regressions[:, 2] * w
    boxes[:, 3] = boxes[:, 3] + regressions[:, 3] * h
    return boxes


def square_boxes(boxes):
    """Convert bounding boxes to square for network input"""
    if len(boxes) == 0:
        return boxes
    
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    max_side = np.maximum(w, h)
    
    boxes[:, 0] = boxes[:, 0] + w * 0.5 - max_side * 0.5
    boxes[:, 1] = boxes[:, 1] + h * 0.5 - max_side * 0.5
    boxes[:, 2] = boxes[:, 0] + max_side
    boxes[:, 3] = boxes[:, 1] + max_side
    return boxes


def crop_and_pad(img, boxes, target_size):
    """Crop regions from image and resize to target_size (with zero padding on boundaries)"""
    num_boxes = len(boxes)
    if num_boxes == 0:
        return np.empty((0, 3, target_size, target_size), dtype=np.float32)
    
    h_img, w_img = img.shape[:2]
    crops = []
    
    for box in boxes:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # Calculate padding if box extends beyond image bounds
        pad_x1 = max(0, -x1)
        pad_y1 = max(0, -y1)
        pad_x2 = max(0, x2 - w_img + 1)
        pad_y2 = max(0, y2 - h_img + 1)
        
        crop_x1 = max(0, x1)
        crop_y1 = max(0, y1)
        crop_x2 = min(w_img - 1, x2)
        crop_y2 = min(h_img - 1, y2)
        
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            crop = np.zeros((target_size, target_size, 3), dtype=np.float32)
        else:
            cropped = img[crop_y1:crop_y2 + 1, crop_x1:crop_x2 + 1]
            if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
                cropped = cv2.copyMakeBorder(
                    cropped, pad_y1, pad_y2, pad_x1, pad_x2,
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            crop = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            
        # Normalize to [-1, 1] as expected by MTCNN
        crop = (crop.astype(np.float32) - 127.5) * 0.0078125
        crops.append(crop.transpose(2, 0, 1))  # HWC -> CHW
        
    return np.ascontiguousarray(np.stack(crops), dtype=np.float32)


class OpenVINOMTCNN:
    """
    MTCNN implementation running on Intel OpenVINO FP16 runtime.
    Features:
    - P-Net (Proposal Network) with dynamic shape support
    - R-Net (Refinement Network) with batched inference
    - O-Net (Output Network) with facial landmark regression (5 points)
    - Warp Affine Facial Alignment
    """
    def __init__(self, model_dir=None, min_face_size=30, scale_factor=0.709,
                 thresholds=(0.6, 0.7, 0.8), device="CPU"):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "models", "openvino_fp16")
            
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.thresholds = thresholds
        
        self.core = ov.Core()
        
        # Load and compile models
        pnet_xml = os.path.join(model_dir, "pnet.xml")
        rnet_xml = os.path.join(model_dir, "rnet.xml")
        onet_xml = os.path.join(model_dir, "onet.xml")
        
        config = {
            ov.properties.hint.performance_mode(): ov.properties.hint.PerformanceMode.LATENCY,
            ov.properties.hint.execution_mode(): ov.properties.hint.ExecutionMode.PERFORMANCE
        }
        
        self.pnet_compiled = self.core.compile_model(pnet_xml, device, config)
        self.rnet_compiled = self.core.compile_model(rnet_xml, device, config)
        self.onet_compiled = self.core.compile_model(onet_xml, device, config)
        
        self.pnet_req = self.pnet_compiled.create_infer_request()
        self.rnet_req = self.rnet_compiled.create_infer_request()
        self.onet_req = self.onet_compiled.create_infer_request()
        
        print(f"[OpenVINOMTCNN] Loaded FP16 MTCNN models on {device}")

    def _generate_scales(self, h, w):
        scales = []
        m = 12.0 / self.min_face_size
        min_dim = min(h, w) * m
        
        factor = self.scale_factor
        factor_count = 0
        while min_dim >= 12.0:
            scales.append(m * (factor ** factor_count))
            min_dim *= factor
            factor_count += 1
        return scales

    def _stage_pnet(self, rgb_img):
        h, w = rgb_img.shape[:2]
        scales = self._generate_scales(h, w)
        
        total_boxes = []
        
        for scale in scales:
            sw, sh = int(np.ceil(w * scale)), int(np.ceil(h * scale))
            if sw < 12 or sh < 12:
                continue
                
            resized = cv2.resize(rgb_img, (sw, sh), interpolation=cv2.INTER_LINEAR)
            # Normalize to [-1, 1]
            inp = (resized.astype(np.float32) - 127.5) * 0.0078125
            inp = np.ascontiguousarray(inp.transpose(2, 0, 1)[None, ...], dtype=np.float32)
            
            # OpenVINO P-Net inference
            res = self.pnet_req.infer({0: inp})
            # Outputs: probabilities and regressions
            outputs = list(res.values())
            
            # Find which output is prob (channel 2) vs reg (channel 4)
            if outputs[0].shape[1] == 2:
                probs = outputs[0][0, 1, :, :]
                regs = outputs[1][0]
            else:
                probs = outputs[1][0, 1, :, :]
                regs = outputs[0][0]
                
            stride = 2
            cell_size = 12
            
            mask = probs >= self.thresholds[0]
            y_idx, x_idx = np.nonzero(mask)
            
            if len(y_idx) == 0:
                continue
                
            scores = probs[y_idx, x_idx]
            reg = regs[:, y_idx, x_idx].T  # shape (N, 4)
            
            b_x1 = np.round((stride * x_idx) / scale)
            b_y1 = np.round((stride * y_idx) / scale)
            b_x2 = np.round((stride * x_idx + cell_size) / scale)
            b_y2 = np.round((stride * y_idx + cell_size) / scale)
            
            boxes = np.column_stack([b_x1, b_y1, b_x2, b_y2, scores, reg])
            
            keep = nms(boxes[:, :5], 0.5, "Union")
            if len(keep) > 0:
                total_boxes.append(boxes[keep])
                
        if len(total_boxes) == 0:
            return np.empty((0, 9), dtype=np.float32)
            
        total_boxes = np.vstack(total_boxes)
        keep = nms(total_boxes[:, :5], 0.7, "Union")
        total_boxes = total_boxes[keep]
        
        boxes = calibrate_boxes(total_boxes[:, :5].copy(), total_boxes[:, 5:])
        boxes = square_boxes(boxes)
        return boxes

    def _stage_rnet(self, rgb_img, boxes):
        if len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
            
        crops = crop_and_pad(rgb_img, boxes, 24)
        if len(crops) == 0:
            return np.empty((0, 5), dtype=np.float32)
            
        res = self.rnet_req.infer({0: crops})
        outputs = list(res.values())
        
        if outputs[0].shape[1] == 2:
            probs = outputs[0][:, 1]
            regs = outputs[1]
        else:
            probs = outputs[1][:, 1]
            regs = outputs[0]
            
        keep = np.where(probs >= self.thresholds[1])[0]
        if len(keep) == 0:
            return np.empty((0, 5), dtype=np.float32)
            
        boxes = boxes[keep]
        boxes[:, 4] = probs[keep]
        regs = regs[keep]
        
        keep_nms = nms(boxes, 0.7, "Union")
        boxes = boxes[keep_nms]
        regs = regs[keep_nms]
        
        boxes = calibrate_boxes(boxes, regs)
        boxes = square_boxes(boxes)
        return boxes

    def _stage_onet(self, rgb_img, boxes):
        if len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)
            
        crops = crop_and_pad(rgb_img, boxes, 48)
        if len(crops) == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)
            
        res = self.onet_req.infer({0: crops})
        outputs = list(res.values())
        
        # Sort outputs by channel count: 2 (prob), 4 (box reg), 10 (landmarks)
        prob_out = next(o for o in outputs if o.shape[1] == 2)
        reg_out = next(o for o in outputs if o.shape[1] == 4)
        lm_out = next(o for o in outputs if o.shape[1] == 10)
        
        probs = prob_out[:, 1]
        regs = reg_out
        landmarks = lm_out
        
        keep = np.where(probs >= self.thresholds[2])[0]
        if len(keep) == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)
            
        boxes = boxes[keep]
        boxes[:, 4] = probs[keep]
        regs = regs[keep]
        landmarks = landmarks[keep]
        
        # Compute absolute facial landmark coordinates
        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1
        
        pts = np.zeros((len(boxes), 5, 2), dtype=np.float32)
        # Landmark layout: x0-x4 are first 5, y0-y4 are next 5
        for p in range(5):
            pts[:, p, 0] = boxes[:, 0] + landmarks[:, p] * w
            pts[:, p, 1] = boxes[:, 1] + landmarks[:, p + 5] * h
            
        boxes = calibrate_boxes(boxes, regs)
        keep_nms = nms(boxes, 0.7, "Min")
        
        return boxes[keep_nms], pts[keep_nms]

    def detect(self, bgr_or_rgb_img, is_bgr=True):
        """
        Detect faces in image.
        Returns:
            boxes: array of shape (N, 5) with [x1, y1, x2, y2, score]
            landmarks: array of shape (N, 5, 2) with 5 landmark points (x, y)
        """
        if is_bgr:
            rgb_img = cv2.cvtColor(bgr_or_rgb_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = bgr_or_rgb_img
            
        # Stage 1: P-Net
        boxes = self._stage_pnet(rgb_img)
        if len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)
            
        # Stage 2: R-Net
        boxes = self._stage_rnet(rgb_img, boxes)
        if len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32), np.empty((0, 5, 2), dtype=np.float32)
            
        # Stage 3: O-Net
        boxes, landmarks = self._stage_onet(rgb_img, boxes)
        return boxes, landmarks

    @staticmethod
    def align_face(img, landmarks, target_size=(160, 160)):
        """
        Align face using 5 facial landmarks and similarity transformation (Warp Affine).
        Standard ArcFace / FaceNet reference template for 112x112 or 160x160:
        5 points: left_eye, right_eye, nose, left_mouth, right_mouth
        """
        # Standard reference template points for 112x112 scaled to target_size
        ref_pts_112 = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        
        scale_x = target_size[0] / 112.0
        scale_y = target_size[1] / 112.0
        ref_pts = ref_pts_112.copy()
        ref_pts[:, 0] *= scale_x
        ref_pts[:, 1] *= scale_y
        
        # Estimate similarity transformation matrix
        src_pts = landmarks.astype(np.float32)
        tfm, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts)
        if tfm is None:
            # Fallback simple crop if affine matrix cannot be estimated
            x1, y1 = np.min(src_pts, axis=0).astype(int)
            x2, y2 = np.max(src_pts, axis=0).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            crop = img[y1:y2, x1:x2]
            return cv2.resize(crop, target_size)
            
        aligned = cv2.warpAffine(
            img, tfm, target_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        return aligned
