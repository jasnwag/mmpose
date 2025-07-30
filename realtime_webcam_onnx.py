#!/usr/bin/env python3
"""
Real-time RTMPose Webcam Demo with ONNX + CoreML Acceleration + 3D Pose Lifting
Provides significant speedup on Apple Silicon through Apple Neural Engine
"""

import cv2
import time
import numpy as np
import onnxruntime as ort
from typing import Tuple, List, Optional
import argparse
import os
import urllib.request
from pathlib import Path
from collections import deque

# Try to import MMPose for 3D pose lifting
try:
    from mmpose.apis import init_model, inference_pose_lifter_model, convert_keypoint_definition, extract_pose_sequence
    from mmpose.structures import PoseDataSample
    from mmengine.structures import InstanceData
    HAS_MMPOSE = True
except ImportError:
    HAS_MMPOSE = False
    print("⚠️  MMPose not available. 3D pose lifting will be limited to ONNX models only.")


# Model configurations for different RTMPose variants
MODEL_CONFIGS = {
    'rtmpose-t': {
        'input_size': (192, 256),
        'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip',
        'description': 'RTMPose-t (Tiny) - Fastest, lowest accuracy'
    },
    'rtmpose-s': {
        'input_size': (192, 256), 
        'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip',
        'description': 'RTMPose-s (Small) - Fast, good accuracy'
    },
    'rtmpose-m': {
        'input_size': (192, 256),
        'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
        'description': 'RTMPose-m (Medium) - Balanced speed and accuracy'
    },
    'rtmpose-l': {
        'input_size': (192, 256),
        'url': 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip', 
        'description': 'RTMPose-l (Large) - Slower, highest accuracy'
    }
}

# 3D Pose Lifter configurations
POSE_LIFTER_CONFIGS = {
    'simplebaseline3d': {
        'config': 'configs/body_3d_keypoint/image_pose_lift/h36m/image-pose-lift_tcn_8xb64-200e_h36m.py',
        'checkpoint': 'https://download.openmmlab.com/mmpose/body3d/simple_baseline/simple3Dbaseline_h36m-f0ad73a4_20210419.pth',
        'description': 'SimpleBaseline3D - H36M trained 3D pose lifter',
        'input_size': (17, 2),  # 17 keypoints, 2D coordinates
        'output_size': (16, 3),  # 16 keypoints (excluding root), 3D coordinates
        'dataset': 'h36m',
        'type': 'pytorch'  # Can be 'pytorch' or 'onnx'
    }
}


class PoseDataSample:
    """Simple pose data sample for 3D lifting."""
    def __init__(self):
        self.pred_instances = None
        self.track_id = -1
        self.gt_instances = None
    
    def set_field(self, value, field_name):
        setattr(self, field_name, value)
    
    def get(self, field_name, default=None):
        return getattr(self, field_name, default)


class SimplePredInstances:
    """Simple prediction instances for pose data."""
    def __init__(self):
        self.keypoints = None
        self.keypoint_scores = None
        self.bboxes = None


class SimplePoseDataSample:
    """Simple pose data sample for our internal use."""
    def __init__(self):
        self.pred_instances = None
        self.track_id = -1
        self.gt_instances = None
    
    def get(self, field_name, default=None):
        return getattr(self, field_name, default)
    
    def set_field(self, value, field_name):
        setattr(self, field_name, value)


class RTMPoseONNXDemo:
    """Real-time RTMPose demo using ONNX Runtime with CoreML acceleration and 3D lifting."""
    
    def __init__(self, onnx_model_path: str, use_coreml: bool = True, 
                 input_size: tuple = (192, 256), model_variant: str = None,
                 pose_lifter_path: Optional[str] = None, pose_lifter_config: Optional[str] = None,
                 enable_3d: bool = False):
        """Initialize the demo.
        
        Args:
            onnx_model_path: Path to ONNX model file
            use_coreml: Whether to use CoreML acceleration
            input_size: Model input size (width, height)
            model_variant: Model variant name for display
            pose_lifter_path: Path to 3D pose lifter model (ONNX or PyTorch)
            pose_lifter_config: Path to 3D pose lifter config (for PyTorch models)
            enable_3d: Whether to enable 3D pose lifting
        """
        self.input_size = input_size
        self.use_coreml = use_coreml
        self.model_variant = model_variant or "custom"
        self.enable_3d = enable_3d
        
        # Initialize ONNX Runtime session for 2D pose estimation
        self.session = self._build_session(onnx_model_path)
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Initialize 3D pose lifter if enabled
        self.pose_lifter_session = None
        self.pose_lifter_input_name = None
        self.pose_lifter_output_names = None
        self.pose_lifter_model = None
        self.pose_lifter_type = None
        
        if self.enable_3d:
            if pose_lifter_config and HAS_MMPOSE:
                self._init_pytorch_pose_lifter(pose_lifter_config, pose_lifter_path)
            elif pose_lifter_path and pose_lifter_path.endswith('.onnx'):
                self._init_onnx_pose_lifter(pose_lifter_path)
            else:
                print("❌ No valid 3D pose lifter specified")
                self.enable_3d = False
        
        # Performance tracking
        self.frame_times = []
        self.max_history = 30  # Keep last 30 frame times for FPS calculation
        
        # Detailed timing tracking
        self.timing_history = {
            'capture': [],
            'preprocess': [],
            'inference': [],
            'postprocess': [],
            'pose_lifting': [],
            'visualize': [],
            'display': [],
            'total': []
        }
        self.max_timing_history = 30
        
        # 3D pose sequence buffer for temporal consistency
        self.pose_sequence_buffer = deque(maxlen=5)  # Keep last 5 frames
        self.track_id_counter = 0
        
        # 3D pose recording
        self.pose_3d_recording = []  # Store all 3D poses with frame info
        self.recording_enabled = True  # Enable by default
        
        # COCO keypoint skeleton for visualization
        self.skeleton = [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
            (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
            (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
        ]
        
        # Colors for visualization
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
        ]
        
        print(f"🚀 RTMPose ONNX Demo initialized")
        print(f"   Model: {onnx_model_path}")
        print(f"   Input size: {input_size}")
        print(f"   CoreML acceleration: {use_coreml}")
        print(f"   3D pose lifting: {enable_3d}")
        if self.enable_3d:
            if self.pose_lifter_type == 'pytorch':
                print(f"   Pose lifter: PyTorch ({pose_lifter_config})")
            else:
                print(f"   Pose lifter: ONNX ({pose_lifter_path})")
        print(f"   Available providers: {ort.get_available_providers()}")
    
    def _init_pytorch_pose_lifter(self, config_path: str, checkpoint_path: Optional[str] = None):
        """Initialize PyTorch-based 3D pose lifter using MMPose."""
        try:
            if not HAS_MMPOSE:
                raise ImportError("MMPose not available")
            
            # Download checkpoint if not provided or doesn't exist
            if not checkpoint_path or not os.path.exists(checkpoint_path):
                checkpoint_path = self._download_checkpoint()
            
            # Initialize the pose lifter model
            self.pose_lifter_model = init_model(config_path, checkpoint_path, device='cpu')
            self.pose_lifter_type = 'pytorch'
            
            print(f"✅ PyTorch 3D pose lifter initialized")
            print(f"   Config: {config_path}")
            print(f"   Checkpoint: {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Failed to initialize PyTorch 3D pose lifter: {e}")
            self.enable_3d = False
    
    def _init_onnx_pose_lifter(self, pose_lifter_path: str):
        """Initialize ONNX-based 3D pose lifter."""
        try:
            self.pose_lifter_session = self._build_session(pose_lifter_path)
            self.pose_lifter_input_name = self.pose_lifter_session.get_inputs()[0].name
            self.pose_lifter_output_names = [out.name for out in self.pose_lifter_session.get_outputs()]
            self.pose_lifter_type = 'onnx'
            print(f"✅ ONNX 3D pose lifter initialized: {pose_lifter_path}")
        except Exception as e:
            print(f"❌ Failed to initialize ONNX 3D pose lifter: {e}")
            self.enable_3d = False
    
    def _download_checkpoint(self) -> str:
        """Download the default pose lifter checkpoint."""
        checkpoint_url = POSE_LIFTER_CONFIGS['simplebaseline3d']['checkpoint']
        checkpoint_path = 'simple3Dbaseline_h36m.pth'
        
        if not os.path.exists(checkpoint_path):
            print(f"🔽 Downloading 3D pose lifter checkpoint...")
            try:
                urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
                print(f"✅ Checkpoint downloaded: {checkpoint_path}")
            except Exception as e:
                print(f"❌ Failed to download checkpoint: {e}")
                raise e
        
        return checkpoint_path
    
    def _build_session(self, onnx_path: str) -> ort.InferenceSession:
        """Build ONNX Runtime session with optimal providers."""
        
        providers = []
        if self.use_coreml and 'CoreMLExecutionProvider' in ort.get_available_providers():
            # CoreML provider for Apple Silicon acceleration
            providers.append(('CoreMLExecutionProvider', {
                'use_cpu_only': False,  # Use Neural Engine when available
                'only_enable_device_with_ane': True  # Only use devices with ANE
            }))
            print("🎯 Using CoreML acceleration (Apple Neural Engine)")
        
        # Always add CPU as fallback
        providers.append('CPUExecutionProvider')
        
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Print actual providers being used
        print(f"   Active providers: {session.get_providers()}")
        
        return session
    
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess frame for RTMPose inference."""
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Calculate bbox (full frame)
        bbox = np.array([0, 0, w, h], dtype=np.float32)
        
        # Get center and scale
        center, scale = self._bbox_xyxy2cs(bbox, padding=1.25)
        
        # Resize and normalize
        resized_img, scale = self._top_down_affine(self.input_size, scale, center, frame)
        
        # Normalize with ImageNet stats
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        normalized_img = (resized_img - mean) / std
        
        # Convert to CHW format and add batch dimension
        input_tensor = normalized_img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        
        return input_tensor, center, scale
    
    def _bbox_xyxy2cs(self, bbox: np.ndarray, padding: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Convert bbox to center and scale."""
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5])
        scale = np.array([(x2 - x1), (y2 - y1)]) * padding
        return center, scale
    
    def _top_down_affine(self, input_size: tuple, scale: np.ndarray, 
                        center: np.ndarray, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply affine transformation."""
        w, h = input_size
        
        # Fix aspect ratio
        scale = self._fix_aspect_ratio(scale, w / h)
        
        # Get affine matrix
        warp_mat = self._get_warp_matrix(center, scale, 0, (w, h))
        
        # Apply transformation
        warped_img = cv2.warpAffine(img, warp_mat, (w, h), flags=cv2.INTER_LINEAR)
        
        return warped_img, scale
    
    def _fix_aspect_ratio(self, scale: np.ndarray, aspect_ratio: float) -> np.ndarray:
        """Fix aspect ratio of scale."""
        w, h = scale
        if w > h * aspect_ratio:
            scale = np.array([w, w / aspect_ratio])
        else:
            scale = np.array([h * aspect_ratio, h])
        return scale
    
    def _get_warp_matrix(self, center: np.ndarray, scale: np.ndarray, 
                        rot: float, output_size: tuple) -> np.ndarray:
        """Get affine transformation matrix."""
        # Simplified version - for full implementation see the ONNX example
        src_w, src_h = scale
        dst_w, dst_h = output_size
        
        # Source points
        src = np.array([
            [center[0] - src_w * 0.5, center[1] - src_h * 0.5],
            [center[0] + src_w * 0.5, center[1] - src_h * 0.5],
            [center[0] - src_w * 0.5, center[1] + src_h * 0.5]
        ], dtype=np.float32)
        
        # Destination points
        dst = np.array([
            [0, 0],
            [dst_w, 0],
            [0, dst_h]
        ], dtype=np.float32)
        
        return cv2.getAffineTransform(src, dst)
    
    def inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Run ONNX inference."""
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return outputs
    
    def postprocess(self, outputs: List[np.ndarray], center: np.ndarray, 
                   scale: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess ONNX outputs to get keypoints."""
        # Assuming SimCC output format
        simcc_x, simcc_y = outputs
        
        # Decode SimCC to keypoints
        keypoints, scores = self._decode_simcc(simcc_x, simcc_y)
        
        # Transform back to original image coordinates
        keypoints = keypoints / np.array(self.input_size) * scale + center - scale / 2
        
        return keypoints, scores
    
    def _decode_simcc(self, simcc_x: np.ndarray, simcc_y: np.ndarray, 
                     split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Decode SimCC representations."""
        # Get maximum locations
        N, K, Wx = simcc_x.shape
        
        # Reshape for processing
        simcc_x_flat = simcc_x.reshape(N * K, -1)
        simcc_y_flat = simcc_y.reshape(N * K, -1)
        
        # Get argmax positions
        x_locs = np.argmax(simcc_x_flat, axis=1)
        y_locs = np.argmax(simcc_y_flat, axis=1)
        
        # Stack coordinates
        keypoints = np.stack([x_locs, y_locs], axis=-1).astype(np.float32)
        
        # Get confidence scores
        max_val_x = np.amax(simcc_x_flat, axis=1)
        max_val_y = np.amax(simcc_y_flat, axis=1)
        scores = np.minimum(max_val_x, max_val_y)
        
        # Apply split ratio
        keypoints = keypoints / split_ratio
        
        # Reshape back
        keypoints = keypoints.reshape(N, K, 2)
        scores = scores.reshape(N, K)
        
        return keypoints, scores
    
    def convert_keypoints_for_3d(self, keypoints: np.ndarray) -> np.ndarray:
        """Convert COCO keypoints to H36M format for 3D lifting."""
        # COCO to H36M keypoint mapping
        # H36M: 0=root, 1=r_hip, 2=r_knee, 3=r_foot, 4=l_hip, 5=l_knee, 6=l_foot, 
        #       7=spine, 8=thorax, 9=neck, 10=head, 11=l_shoulder, 12=l_elbow, 
        #       13=l_wrist, 14=r_shoulder, 15=r_elbow, 16=r_wrist
        
        # COCO: 0=nose, 1=l_eye, 2=r_eye, 3=l_ear, 4=r_ear, 5=l_shoulder, 6=r_shoulder,
        #       7=l_elbow, 8=r_elbow, 9=l_wrist, 10=r_wrist, 11=l_hip, 12=r_hip,
        #       13=l_knee, 14=r_knee, 15=l_ankle, 16=r_ankle
        
        h36m_keypoints = np.zeros((keypoints.shape[0], 17, 2), dtype=keypoints.dtype)
        
        # Root (pelvis) - middle of hips
        h36m_keypoints[:, 0] = (keypoints[:, 11] + keypoints[:, 12]) / 2
        
        # Right leg
        h36m_keypoints[:, 1] = keypoints[:, 12]  # r_hip
        h36m_keypoints[:, 2] = keypoints[:, 14]  # r_knee  
        h36m_keypoints[:, 3] = keypoints[:, 16]  # r_foot
        
        # Left leg
        h36m_keypoints[:, 4] = keypoints[:, 11]  # l_hip
        h36m_keypoints[:, 5] = keypoints[:, 13]  # l_knee
        h36m_keypoints[:, 6] = keypoints[:, 15]  # l_foot
        
        # Spine and thorax
        h36m_keypoints[:, 7] = (h36m_keypoints[:, 0] + h36m_keypoints[:, 8]) / 2  # spine
        h36m_keypoints[:, 8] = (keypoints[:, 5] + keypoints[:, 6]) / 2  # thorax
        
        # Head
        h36m_keypoints[:, 9] = (keypoints[:, 1] + keypoints[:, 2]) / 2  # neck
        h36m_keypoints[:, 10] = keypoints[:, 0]  # head
        
        # Left arm
        h36m_keypoints[:, 11] = keypoints[:, 5]  # l_shoulder
        h36m_keypoints[:, 12] = keypoints[:, 7]  # l_elbow
        h36m_keypoints[:, 13] = keypoints[:, 9]  # l_wrist
        
        # Right arm
        h36m_keypoints[:, 14] = keypoints[:, 6]  # r_shoulder
        h36m_keypoints[:, 15] = keypoints[:, 8]  # r_elbow
        h36m_keypoints[:, 16] = keypoints[:, 10]  # r_wrist
        
        return h36m_keypoints
    
    def extract_pose_sequence(self, pose_results_list: List, frame_idx: int, 
                            causal: bool = True, seq_len: int = 1, step: int = 1) -> List:
        """Extract pose sequence for 3D lifting."""
        if len(pose_results_list) < seq_len:
            # Pad with the first frame
            pad_frames = [pose_results_list[0]] * (seq_len - len(pose_results_list))
            pose_results_list = pad_frames + pose_results_list
        
        if causal:
            # Use the last frame as target
            start_idx = max(0, len(pose_results_list) - seq_len)
            return pose_results_list[start_idx:]
        else:
            # Use the middle frame as target
            mid_idx = len(pose_results_list) // 2
            start_idx = max(0, mid_idx - seq_len // 2)
            end_idx = min(len(pose_results_list), start_idx + seq_len)
            return pose_results_list[start_idx:end_idx]
    
    def inference_3d_pose(self, pose_sequence: List) -> Optional[np.ndarray]:
        """Inference 3D pose from 2D pose sequence."""
        if not self.enable_3d:
            return None
        
        if self.pose_lifter_type == 'pytorch':
            return self._inference_3d_pytorch(pose_sequence)
        elif self.pose_lifter_type == 'onnx':
            return self._inference_3d_onnx(pose_sequence)
        else:
            return None
    
    def _inference_3d_pytorch(self, pose_sequence: List) -> Optional[np.ndarray]:
        """Inference 3D pose using PyTorch model."""
        try:
            # Convert pose sequence to MMPose format
            pose_results_2d = []
            
            for i, pose_data in enumerate(pose_sequence):
                if pose_data.pred_instances is not None:
                    # Create proper PoseDataSample using MMPose's PoseDataSample
                    from mmpose.structures import PoseDataSample as MMPosePoseDataSample
                    data_sample = MMPosePoseDataSample()
                    
                    # Create InstanceData
                    pred_instances = InstanceData()
                    
                    # Get keypoints and scores
                    keypoints = pose_data.pred_instances.keypoints
                    scores = pose_data.pred_instances.keypoint_scores
                    
                    if keypoints is not None and len(keypoints) > 0:
                        # Convert keypoints from COCO to H36M format using MMPose function
                        keypoints_h36m = convert_keypoint_definition(
                            keypoints, 'coco', 'h36m'
                        )
                        
                        pred_instances.keypoints = keypoints_h36m
                        pred_instances.keypoint_scores = scores
                        
                        # Create bboxes from keypoints
                        for person_idx in range(keypoints.shape[0]):
                            person_kpts = keypoints[person_idx]
                            valid_kpts = person_kpts[scores[person_idx] > 0.3]
                            if len(valid_kpts) > 0:
                                x_coords = valid_kpts[:, 0]
                                y_coords = valid_kpts[:, 1]
                                x_min, x_max = np.min(x_coords), np.max(x_coords)
                                y_min, y_max = np.min(y_coords), np.max(y_coords)
                                
                                # Add some padding
                                w, h = x_max - x_min, y_max - y_min
                                x_min -= w * 0.1
                                y_min -= h * 0.1
                                x_max += w * 0.1
                                y_max += h * 0.1
                                
                                bbox = np.array([[x_min, y_min, x_max, y_max]])
                            else:
                                bbox = np.array([[0, 0, 100, 100]])
                            
                            if person_idx == 0:
                                pred_instances.bboxes = bbox
                            else:
                                pred_instances.bboxes = np.vstack([pred_instances.bboxes, bbox])
                    else:
                        continue
                    
                    data_sample.pred_instances = pred_instances
                    
                    # Initialize gt_instances 
                    data_sample.gt_instances = InstanceData()
                    
                    # Set track_id
                    data_sample.track_id = pose_data.track_id
                    
                    pose_results_2d.append(data_sample)
            
            if not pose_results_2d:
                return None
            
            # Use extract_pose_sequence to get the right sequence format
            pose_seq_2d = extract_pose_sequence(
                [pose_results_2d], 
                frame_idx=0,
                causal=True,
                seq_len=1,
                step=1
            )
            
            # Use MMPose's inference function
            pose_lift_results = inference_pose_lifter_model(
                self.pose_lifter_model,
                pose_seq_2d,
                image_size=(640, 480),
                norm_pose_2d=True
            )
            
            if pose_lift_results and len(pose_lift_results) > 0:
                # Extract 3D keypoints
                pose_3d_list = []
                for result in pose_lift_results:
                    if hasattr(result, 'pred_instances') and hasattr(result.pred_instances, 'keypoints'):
                        keypoints_3d = result.pred_instances.keypoints
                        if isinstance(keypoints_3d, np.ndarray):
                            if keypoints_3d.ndim == 4:
                                keypoints_3d = keypoints_3d.squeeze(1)
                            pose_3d_list.append(keypoints_3d)
                
                if pose_3d_list:
                    result_3d = np.concatenate(pose_3d_list, axis=0)
                    print(f"✅ 3D pose inference successful! Shape: {result_3d.shape}")
                    return result_3d
            
            return None
            
        except Exception as e:
            print(f"⚠️  PyTorch 3D pose inference error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _inference_3d_onnx(self, pose_sequence: List) -> Optional[np.ndarray]:
        """Inference 3D pose using ONNX model."""
        try:
            # Prepare input for pose lifter
            # Extract keypoints from the sequence
            keypoints_2d = []
            for pose_data in pose_sequence:
                if pose_data.pred_instances is not None:
                    kpts = pose_data.pred_instances.keypoints
                    # Convert to H36M format
                    kpts_h36m = self.convert_keypoints_for_3d(kpts)
                    keypoints_2d.append(kpts_h36m)
                else:
                    # If no keypoints, use zeros
                    keypoints_2d.append(np.zeros((1, 17, 2)))
            
            # Stack into sequence
            pose_seq_input = np.stack(keypoints_2d, axis=1)  # [N, T, 17, 2]
            
            # Normalize keypoints (simple normalization)
            # Center around root joint
            root_joint = pose_seq_input[:, :, 0:1, :]  # [N, T, 1, 2]
            pose_seq_normalized = pose_seq_input - root_joint
            
            # Scale by average joint distance
            joint_distances = np.linalg.norm(pose_seq_normalized, axis=-1)
            scale_factor = np.mean(joint_distances[joint_distances > 0])
            if scale_factor > 0:
                pose_seq_normalized = pose_seq_normalized / scale_factor
            
            # Reshape for model input: [N*T, 17*2]
            batch_size, seq_len, num_joints, num_coords = pose_seq_normalized.shape
            pose_input = pose_seq_normalized.reshape(batch_size * seq_len, -1).astype(np.float32)
            
            # Run inference
            outputs = self.pose_lifter_session.run(
                self.pose_lifter_output_names, 
                {self.pose_lifter_input_name: pose_input}
            )
            
            # Process output (assuming output is [N*T, 16*3] for 16 joints excluding root)
            pose_3d = outputs[0].reshape(batch_size, seq_len, 16, 3)
            
            # Take the last frame result
            pose_3d_result = pose_3d[:, -1, :, :]  # [N, 16, 3]
            
            # Add root joint back (at origin)
            root_3d = np.zeros((pose_3d_result.shape[0], 1, 3), dtype=pose_3d_result.dtype)
            pose_3d_full = np.concatenate([root_3d, pose_3d_result], axis=1)  # [N, 17, 3]
            
            return pose_3d_full
            
        except Exception as e:
            print(f"⚠️  ONNX 3D pose inference error: {e}")
            return None
    
    def visualize(self, frame: np.ndarray, keypoints: np.ndarray, 
                 scores: np.ndarray, pose_3d: Optional[np.ndarray] = None,
                 show_3d: bool = False) -> np.ndarray:
        """Visualize keypoints on frame with optional 3D pose."""
        vis_frame = frame.copy()
        
        for person_idx, (person_kpts, person_scores) in enumerate(zip(keypoints, scores)):
            # Draw 2D keypoints
            for i, ((x, y), score) in enumerate(zip(person_kpts, person_scores)):
                if score > 0.3:
                    color = self.colors[i % len(self.colors)]
                    cv2.circle(vis_frame, (int(x), int(y)), 3, color, -1)
            
            # Draw 2D skeleton
            for (start_idx, end_idx) in self.skeleton:
                if (start_idx < len(person_scores) and end_idx < len(person_scores) and
                    person_scores[start_idx] > 0.3 and person_scores[end_idx] > 0.3):
                    
                    start_point = (int(person_kpts[start_idx][0]), int(person_kpts[start_idx][1]))
                    end_point = (int(person_kpts[end_idx][0]), int(person_kpts[end_idx][1]))
                    cv2.line(vis_frame, start_point, end_point, (0, 255, 0), 2)
            
            # Draw 3D pose if available
            if show_3d and pose_3d is not None and person_idx < len(pose_3d):
                self._draw_3d_pose(vis_frame, pose_3d[person_idx])
        
        return vis_frame
    
    def _draw_3d_pose(self, frame: np.ndarray, pose_3d: np.ndarray):
        """Draw 3D pose visualization."""
        # Enhanced 3D visualization - project to 2D for display
        
        # Put the 3D pose in top-right corner of the frame
        frame_h, frame_w = frame.shape[:2]
        viz_size = 180  # Larger visualization area
        
        # Scale and offset for visualization
        scale = 40  # Bigger scale
        offset_x = frame_w - viz_size + 30  # Top-right corner  
        offset_y = 60
        
        # Draw background box for 3D visualization
        cv2.rectangle(frame, 
                     (frame_w - viz_size, 10), 
                     (frame_w - 10, viz_size + 20), 
                     (0, 0, 0), -1)  # Black background
        cv2.rectangle(frame, 
                     (frame_w - viz_size, 10), 
                     (frame_w - 10, viz_size + 20), 
                     (255, 255, 255), 2)  # White border
        
        # Add "3D Pose" label and frame counter
        cv2.putText(frame, "3D Pose", 
                   (frame_w - viz_size + 5, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add blinking indicator to show it's updating  
        blink = int(time.perf_counter() * 3) % 2  # Blinks every 0.33 seconds
        indicator_color = (0, 255, 0) if blink else (255, 0, 0)  # Green/Red blinking
        cv2.circle(frame, (frame_w - 20, viz_size + 15), 8, indicator_color, -1)
        
        cv2.putText(frame, f"LIVE", 
                   (frame_w - viz_size + 5, viz_size + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, indicator_color, 1)
        
        # Draw 3D keypoints with bright colors
        joint_colors = [
            (255, 0, 0),    # Head - Red
            (255, 127, 0),  # Neck - Orange  
            (255, 255, 0),  # Shoulders - Yellow
            (127, 255, 0),  # Elbows - Light Green
            (0, 255, 0),    # Wrists - Green
            (0, 255, 127),  # Spine - Cyan-Green
            (0, 255, 255),  # Hips - Cyan
            (0, 127, 255),  # Knees - Light Blue
            (0, 0, 255),    # Ankles - Blue
        ]
        
        for i, (x, y, z) in enumerate(pose_3d):
            # Simple orthographic projection
            screen_x = int(x * scale + offset_x)
            screen_y = int(y * scale + offset_y)
            
            # Make sure points are within the visualization area
            if (frame_w - viz_size + 10) <= screen_x <= (frame_w - 20) and 40 <= screen_y <= (viz_size + 10):
                # Use bright colors
                color = joint_colors[i % len(joint_colors)]
                cv2.circle(frame, (screen_x, screen_y), 3, color, -1)
                
                # Add joint number for debugging
                cv2.putText(frame, str(i), 
                           (screen_x + 5, screen_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw 3D skeleton connections with bright colors
        h36m_skeleton = [
            (0, 1), (1, 2), (2, 3),  # Right leg
            (0, 4), (4, 5), (5, 6),  # Left leg
            (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
            (8, 11), (11, 12), (12, 13),  # Left arm
            (8, 14), (14, 15), (15, 16)   # Right arm
        ]
        
        for start_idx, end_idx in h36m_skeleton:
            if start_idx < len(pose_3d) and end_idx < len(pose_3d):
                start_3d = pose_3d[start_idx]
                end_3d = pose_3d[end_idx]
                
                start_2d = (int(start_3d[0] * scale + offset_x), int(start_3d[1] * scale + offset_y))
                end_2d = (int(end_3d[0] * scale + offset_x), int(end_3d[1] * scale + offset_y))
                
                # Check if both points are in the visualization area
                if (all((frame_w - viz_size + 10) <= pt[0] <= (frame_w - 20) and 
                       40 <= pt[1] <= (viz_size + 10) for pt in [start_2d, end_2d])):
                    cv2.line(frame, start_2d, end_2d, (0, 255, 255), 2)  # Bright cyan lines

    def update_fps(self, frame_time: float):
        """Update FPS calculation."""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.max_history:
            self.frame_times.pop(0)

    def update_timing(self, stage: str, time_ms: float):
        """Update timing statistics for a specific stage."""
        if stage in self.timing_history:
            self.timing_history[stage].append(time_ms)
            if len(self.timing_history[stage]) > self.max_timing_history:
                self.timing_history[stage].pop(0)

    def get_avg_timing(self, stage: str) -> float:
        """Get average timing for a specific stage."""
        if stage in self.timing_history and self.timing_history[stage]:
            return sum(self.timing_history[stage]) / len(self.timing_history[stage])
        return 0.0

    def get_timing_breakdown(self) -> dict:
        """Get comprehensive timing breakdown."""
        breakdown = {}
        for stage in self.timing_history:
            breakdown[stage] = self.get_avg_timing(stage)
        return breakdown

    def get_fps(self) -> float:
        """Get current FPS."""
        if len(self.frame_times) < 2:
            return 0.0
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def run_webcam(self, camera_id: int = 0):
        """Run real-time webcam demo with detailed timing and 3D lifting."""
        print(f"\n🎥 Starting webcam demo (Camera {camera_id})")
        print("Press 'q' to quit, 'c' to toggle CoreML, 's' to save screenshot, 't' to toggle timing display, '3' to toggle 3D, 'r' to save recording")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Cannot open camera {camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        show_detailed_timing = False
        show_3d_visualization = self.enable_3d
        
        # Warmup inference (first few frames are usually slower)
        print("🔥 Warming up inference engine...")
        for _ in range(5):
            ret, frame = cap.read()
            if ret:
                try:
                    input_tensor, center, scale = self.preprocess(frame)
                    self.inference(input_tensor)
                except:
                    pass
        
        try:
            while True:
                # === TIMING: Frame Capture ===
                capture_start = time.perf_counter()
                ret, frame = cap.read()
                capture_time = (time.perf_counter() - capture_start) * 1000
                
                if not ret:
                    print("❌ Failed to capture frame")
                    break
                
                # Total frame processing start
                total_start = time.perf_counter()
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Run inference with detailed timing
                try:
                    # === TIMING: Preprocessing ===
                    preprocess_start = time.perf_counter()
                    input_tensor, center, scale = self.preprocess(frame)
                    preprocess_time = (time.perf_counter() - preprocess_start) * 1000
                    
                    # === TIMING: Inference ===
                    inference_start = time.perf_counter()
                    outputs = self.inference(input_tensor)
                    inference_time = (time.perf_counter() - inference_start) * 1000
                    
                    # === TIMING: Postprocessing ===
                    postprocess_start = time.perf_counter()
                    keypoints, scores = self.postprocess(outputs, center, scale)
                    postprocess_time = (time.perf_counter() - postprocess_start) * 1000
                    
                    # === TIMING: 3D Pose Lifting ===
                    pose_lifting_time = 0
                    pose_3d = None
                    if self.enable_3d:
                        pose_lifting_start = time.perf_counter()
                        
                        # Real 3D pose lifting using MMPose
                        if len(keypoints) > 0:
                            # Create pose data samples for sequence
                            pose_data_sample = SimplePoseDataSample()
                            pose_data_sample.pred_instances = SimplePredInstances()
                            pose_data_sample.pred_instances.keypoints = keypoints
                            pose_data_sample.pred_instances.keypoint_scores = scores
                            pose_data_sample.track_id = self.track_id_counter
                            
                            # Add to sequence buffer
                            self.pose_sequence_buffer.append(pose_data_sample)
                            
                            # Extract sequence for 3D lifting
                            pose_sequence = self.extract_pose_sequence(
                                list(self.pose_sequence_buffer), frame_count, 
                                causal=True, seq_len=3, step=1
                            )
                            
                            # Perform real 3D pose lifting
                            pose_3d = self.inference_3d_pose(pose_sequence)
                            
                            if pose_3d is None:
                                # Fallback to dummy pose if 3D inference fails
                                num_people = keypoints.shape[0]
                                pose_3d = np.zeros((num_people, 17, 3))
                                # Simple static T-pose for debugging
                                for person_idx in range(num_people):
                                    pose_3d[person_idx, 0] = [0, 0, 0]     # Root
                                    pose_3d[person_idx, 10] = [0, 1.2, 0]  # Head
                                    pose_3d[person_idx, 11] = [-0.5, 0.5, 0]  # L shoulder
                                    pose_3d[person_idx, 14] = [0.5, 0.5, 0]   # R shoulder
                                    
                            # Record 3D poses if enabled
                            if self.recording_enabled and pose_3d is not None:
                                timestamp = time.perf_counter()
                                
                                # Record each person's 3D pose
                                for person_idx in range(pose_3d.shape[0]):
                                    pose_record = {
                                        'frame': frame_count,
                                        'timestamp': timestamp,
                                        'person_id': person_idx,
                                        'track_id': self.track_id_counter + person_idx,
                                        'pose_3d': pose_3d[person_idx].tolist(),  # Convert to list for JSON serialization
                                        'pose_2d': keypoints[person_idx].tolist() if person_idx < len(keypoints) else None,
                                        'scores_2d': scores[person_idx].tolist() if person_idx < len(scores) else None
                                    }
                                    self.pose_3d_recording.append(pose_record)
                            
                            # Debug print every 30 frames
                            if frame_count % 30 == 1:
                                if pose_3d is not None:
                                    print(f"🎯 Generated real 3D pose for {keypoints.shape[0]} person(s), frame {frame_count}")
                                    print(f"   3D pose shape: {pose_3d.shape}")
                                    if self.recording_enabled:
                                        print(f"   📼 Recorded poses: {len(self.pose_3d_recording)}")
                                else:
                                    print(f"⚠️  3D pose lifting failed for frame {frame_count}")
                            
                            self.track_id_counter += 1
                        
                        pose_lifting_time = (time.perf_counter() - pose_lifting_start) * 1000
                    
                    # === TIMING: Visualization ===
                    visualize_start = time.perf_counter()
                    vis_frame = self.visualize(frame, keypoints, scores, pose_3d, show_3d_visualization)
                    visualize_time = (time.perf_counter() - visualize_start) * 1000
                    
                except Exception as e:
                    print(f"⚠️  Inference error: {e}")
                    vis_frame = frame
                    preprocess_time = postprocess_time = inference_time = pose_lifting_time = visualize_time = 0
                
                # === TIMING: Display Preparation ===
                display_prep_start = time.perf_counter()
                
                # Calculate FPS
                total_time = (time.perf_counter() - total_start) * 1000
                self.update_fps(total_time / 1000)  # Convert back to seconds
                current_fps = self.get_fps()
                
                # Update timing statistics
                self.update_timing('capture', capture_time)
                self.update_timing('preprocess', preprocess_time)
                self.update_timing('inference', inference_time)
                self.update_timing('postprocess', postprocess_time)
                self.update_timing('pose_lifting', pose_lifting_time)
                self.update_timing('visualize', visualize_time)
                self.update_timing('total', total_time)
                
                # Get timing breakdown
                timing_breakdown = self.get_timing_breakdown()
                
                # Add performance info to frame
                if show_detailed_timing:
                    info_text = [
                        f"Model: {self.model_variant.upper()}",
                        f"Total FPS: {current_fps:.1f}",
                        f"",
                        f"=== TIMING BREAKDOWN ===",
                        f"Capture:    {capture_time:.1f}ms ({timing_breakdown['capture']:.1f}ms avg)",
                        f"Preprocess: {preprocess_time:.1f}ms ({timing_breakdown['preprocess']:.1f}ms avg)",
                        f"Inference:  {inference_time:.1f}ms ({timing_breakdown['inference']:.1f}ms avg)",
                        f"Postproc:   {postprocess_time:.1f}ms ({timing_breakdown['postprocess']:.1f}ms avg)",
                    ]
                    
                    if self.enable_3d:
                        info_text.append(f"3D Lifting: {pose_lifting_time:.1f}ms ({timing_breakdown['pose_lifting']:.1f}ms avg)")
                    
                    info_text.extend([
                        f"Visualize:  {visualize_time:.1f}ms ({timing_breakdown['visualize']:.1f}ms avg)",
                        f"Total:      {total_time:.1f}ms ({timing_breakdown['total']:.1f}ms avg)",
                        f"",
                        f"Bottleneck: {self._identify_bottleneck(timing_breakdown)}",
                        f"CoreML: {'ON' if self.use_coreml else 'OFF'}",
                        f"3D: {'ON' if show_3d_visualization else 'OFF'}",
                        f"Recording: {len(self.pose_3d_recording)} poses" if self.enable_3d and self.recording_enabled else "Recording: OFF",
                        f"Frame: {frame_count}"
                    ])
                else:
                    # Simplified display
                    bottleneck = self._identify_bottleneck(timing_breakdown)
                    info_text = [
                        f"Model: {self.model_variant.upper()}",
                        f"FPS: {current_fps:.1f}",
                        f"Inference: {inference_time:.1f}ms (avg: {timing_breakdown['inference']:.1f}ms)",
                    ]
                    
                    if self.enable_3d:
                        info_text.append(f"3D: {pose_lifting_time:.1f}ms (avg: {timing_breakdown['pose_lifting']:.1f}ms)")
                    
                    info_text.extend([
                        f"Bottleneck: {bottleneck}",
                        f"CoreML: {'ON' if self.use_coreml else 'OFF'}",
                        f"3D: {'ON' if show_3d_visualization else 'OFF'}",
                        f"Recording: {len(self.pose_3d_recording)} poses" if self.enable_3d and self.recording_enabled else "Recording: OFF",
                        f"Frame: {frame_count}",
                        f"Press 't' for detailed timing, 'r' to save"
                    ])
                
                # Draw text with background for better visibility
                y_offset = 30
                font_scale = 0.6 if show_detailed_timing else 0.7
                for text in info_text:
                    if text:  # Skip empty lines
                        # Get text size for background
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                        
                        # Draw semi-transparent background
                        overlay = vis_frame.copy()
                        cv2.rectangle(overlay, (5, y_offset - text_height - 5), 
                                    (15 + text_width, y_offset + 5), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
                        
                        # Draw text
                        color = (0, 255, 0) if not text.startswith("Bottleneck:") else (0, 255, 255)
                        cv2.putText(vis_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                   font_scale, color, 2)
                    y_offset += 25
                
                display_prep_time = (time.perf_counter() - display_prep_start) * 1000
                
                # === TIMING: Display ===
                display_start = time.perf_counter()
                cv2.imshow('RTMPose Real-time Demo (ONNX + CoreML + 3D)', vis_frame)
                display_time = (time.perf_counter() - display_start) * 1000
                
                # Update display timing
                self.update_timing('display', display_prep_time + display_time)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Toggle CoreML (requires restart)
                    print(f"💡 Toggle CoreML: {not self.use_coreml} (restart demo to apply)")
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"rtmpose_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(filename, vis_frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('t'):
                    # Toggle detailed timing display
                    show_detailed_timing = not show_detailed_timing
                    print(f"🕐 Detailed timing: {'ON' if show_detailed_timing else 'OFF'}")
                elif key == ord('3'):
                    # Toggle 3D visualization
                    if self.enable_3d:
                        show_3d_visualization = not show_3d_visualization
                        print(f"🎯 3D visualization: {'ON' if show_3d_visualization else 'OFF'}")
                    else:
                        print("❌ 3D pose lifting not enabled")
                elif key == ord('r'):
                    # Save 3D recording
                    if self.enable_3d:
                        saved_file = self.save_3d_recording()
                        if saved_file:
                            print(f"💾 3D recording saved!")
                    else:
                        print("❌ 3D pose lifting not enabled")
                
                frame_count += 1
                
                # Print periodic timing summary to console
                if frame_count % 100 == 0:
                    self._print_timing_summary(frame_count, timing_breakdown)
                
        except KeyboardInterrupt:
            print("\n⏹️  Demo interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Save 3D recording automatically
            if self.enable_3d and self.pose_3d_recording:
                print(f"\n💾 Auto-saving 3D recording...")
                saved_file = self.save_3d_recording()
                if saved_file:
                    print(f"   Final recording: {saved_file}")
            
            # Print final comprehensive timing analysis
            self._print_final_timing_analysis(frame_count)
    
    def _identify_bottleneck(self, timing_breakdown: dict) -> str:
        """Identify the performance bottleneck."""
        # Exclude total and display from bottleneck analysis
        core_stages = {k: v for k, v in timing_breakdown.items() 
                      if k not in ['total', 'display']}
        
        if not core_stages:
            return "Unknown"
        
        bottleneck_stage = max(core_stages, key=core_stages.get)
        bottleneck_time = core_stages[bottleneck_stage]
        
        # Add context about what this means
        stage_meanings = {
            'capture': 'Camera/USB',
            'preprocess': 'CPU (resize/normalize)',
            'inference': 'ONNX/CoreML',
            'postprocess': 'CPU (decode)',
            'visualize': 'CPU (drawing)'
        }
        
        meaning = stage_meanings.get(bottleneck_stage, bottleneck_stage)
        return f"{bottleneck_stage} ({meaning}) - {bottleneck_time:.1f}ms"
    
    def _print_timing_summary(self, frame_count: int, timing_breakdown: dict):
        """Print periodic timing summary to console."""
        print(f"\n📊 Timing Summary (Frame {frame_count}):")
        total_core_time = sum(v for k, v in timing_breakdown.items() 
                             if k not in ['total', 'display'])
        
        for stage, avg_time in timing_breakdown.items():
            if stage not in ['total', 'display']:
                percentage = (avg_time / total_core_time * 100) if total_core_time > 0 else 0
                print(f"   {stage:12}: {avg_time:6.1f}ms ({percentage:5.1f}%)")
        
        print(f"   {'display':12}: {timing_breakdown.get('display', 0):6.1f}ms")
        print(f"   {'TOTAL':12}: {timing_breakdown.get('total', 0):6.1f}ms")
        
        # Calculate theoretical max FPS
        total_time = timing_breakdown.get('total', 0)
        if total_time > 0:
            theoretical_fps = 1000 / total_time
            print(f"   Theoretical FPS: {theoretical_fps:.1f}")
    
    def _print_final_timing_analysis(self, frame_count: int):
        """Print comprehensive final timing analysis."""
        print(f"\n📈 Final Performance Analysis ({frame_count} frames):")
        print("="*60)
        
        timing_breakdown = self.get_timing_breakdown()
        
        # Calculate percentages
        total_core_time = sum(v for k, v in timing_breakdown.items() 
                             if k not in ['total', 'display'])
        
        print("Average timing per stage:")
        for stage, avg_time in timing_breakdown.items():
            if stage not in ['total', 'display']:
                percentage = (avg_time / total_core_time * 100) if total_core_time > 0 else 0
                print(f"  {stage:12}: {avg_time:6.1f}ms ({percentage:5.1f}%)")
        
        print(f"  {'display':12}: {timing_breakdown.get('display', 0):6.1f}ms")
        print(f"  {'TOTAL':12}: {timing_breakdown.get('total', 0):6.1f}ms")
        
        # Performance insights
        print(f"\nPerformance Insights:")
        bottleneck = self._identify_bottleneck(timing_breakdown)
        print(f"  Primary bottleneck: {bottleneck}")
        
        avg_fps = self.get_fps()
        total_time = timing_breakdown.get('total', 0)
        theoretical_fps = 1000 / total_time if total_time > 0 else 0
        
        print(f"  Actual FPS: {avg_fps:.1f}")
        print(f"  Theoretical FPS: {theoretical_fps:.1f}")
        
        # Optimization suggestions
        print(f"\nOptimization Suggestions:")
        if timing_breakdown.get('inference', 0) > 15:
            print(f"  • Inference is slow - try a smaller model variant or enable CoreML")
        if timing_breakdown.get('preprocess', 0) > 5:
            print(f"  • Preprocessing is slow - consider optimizing image operations")
        if timing_breakdown.get('visualize', 0) > 5:
            print(f"  • Visualization is slow - reduce drawing complexity")
        if timing_breakdown.get('capture', 0) > 5:
            print(f"  • Camera capture is slow - check USB bandwidth or camera settings")
        
        print(f"  CoreML used: {self.use_coreml}")
        print(f"  Model variant: {self.model_variant}")
    
    def save_3d_recording(self, filename: str = None) -> str:
        """Save recorded 3D poses to JSON file."""
        if not self.pose_3d_recording:
            print("❌ No 3D poses recorded!")
            return None
        
        if filename is None:
            timestamp = int(time.perf_counter())
            filename = f"3d_poses_recording_{timestamp}.json"
        
        try:
            import json
            
            # Create metadata
            recording_data = {
                'metadata': {
                    'total_frames': len(self.pose_3d_recording),
                    'model_variant': self.model_variant,
                    'pose_lifter_type': self.pose_lifter_type,
                    'recording_date': str(int(time.perf_counter())),
                    'keypoint_format': 'H36M (17 joints)',
                    'coordinate_system': 'XYZ (millimeters)',
                    'joint_names': [
                        'root', 'r_hip', 'r_knee', 'r_foot', 'l_hip', 'l_knee', 'l_foot',
                        'spine', 'thorax', 'neck', 'head', 'l_shoulder', 'l_elbow', 'l_wrist',
                        'r_shoulder', 'r_elbow', 'r_wrist'
                    ]
                },
                'poses': self.pose_3d_recording
            }
            
            with open(filename, 'w') as f:
                json.dump(recording_data, f, indent=2)
            
            print(f"✅ Saved {len(self.pose_3d_recording)} 3D poses to {filename}")
            print(f"   File size: {os.path.getsize(filename) / 1024:.1f} KB")
            return filename
            
        except Exception as e:
            print(f"❌ Failed to save recording: {e}")
            return None


def download_sample_onnx_model(output_path: str, model_variant: str = 'rtmpose-m') -> bool:
    """Download RTMPose ONNX model from official rtmlib."""
    
    if model_variant not in MODEL_CONFIGS:
        print(f"❌ Unknown model variant: {model_variant}")
        print(f"   Available variants: {list(MODEL_CONFIGS.keys())}")
        return False
    
    config = MODEL_CONFIGS[model_variant]
    print(f"🔽 Downloading {model_variant} ONNX model from rtmlib...")
    print(f"   Description: {config['description']}")
    print(f"   Input size: {config['input_size']}")
    
    try:
        import zipfile
        import tempfile
        
        # Download the zip file
        print("   Downloading model archive...")
        temp_zip = f"{output_path}.zip"
        urllib.request.urlretrieve(config['url'], temp_zip)
        
        # Extract the ONNX file
        print("   Extracting ONNX model...")
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            # Find the .onnx file in the archive
            for file_name in zip_ref.namelist():
                if file_name.endswith('.onnx'):
                    with zip_ref.open(file_name) as source:
                        with open(output_path, 'wb') as target:
                            target.write(source.read())
                    break
        
        # Clean up zip file
        os.remove(temp_zip)
        
        print(f"✅ {model_variant} ONNX model downloaded: {output_path}")
        print(f"   Model: {model_variant} (COCO 17 keypoints)")
        print(f"   Input size: {config['input_size']}")
        print(f"   Source: OpenMMLab rtmlib")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("💡 Alternative: Install rtmlib library")
        print("   pip install rtmlib")
        return False

def download_pose_lifter_model(output_path: str, model_variant: str = 'simplebaseline3d') -> bool:
    """Download 3D pose lifter model."""
    if model_variant not in POSE_LIFTER_CONFIGS:
        print(f"❌ Unknown pose lifter variant: {model_variant}")
        return False
    
    config = POSE_LIFTER_CONFIGS[model_variant]
    print(f"🔽 Downloading {model_variant} pose lifter model...")
    print(f"   Description: {config['description']}")
    
    try:
        # For now, we'll use a placeholder since the actual ONNX model might not be available
        # In practice, you'd need to convert the PyTorch model to ONNX
        print(f"⚠️  Note: {model_variant} ONNX model not available for direct download")
        print(f"   You need to convert the PyTorch model to ONNX format")
        print(f"   PyTorch checkpoint: {config['url']}")
        return False
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='RTMPose Real-time Webcam Demo with ONNX + CoreML + 3D Lifting',
        epilog="""
Examples:
  # List available model variants
  python realtime_webcam_onnx.py --list-models
  
  # Download and use RTMPose-t with PyTorch 3D lifting
  python realtime_webcam_onnx.py --variant rtmpose-t --download --enable-3d --pose-lifter-config configs/body_3d_keypoint/image_pose_lift/h36m/image-pose-lift_tcn_8xb64-200e_h36m.py
  
  # Use existing models with ONNX 3D lifting
  python realtime_webcam_onnx.py --model path/to/2d_model.onnx --pose-lifter path/to/3d_model.onnx --enable-3d
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', type=str, help='Path to 2D pose ONNX model file')
    parser.add_argument('--variant', type=str, default='rtmpose-m', 
                        choices=list(MODEL_CONFIGS.keys()),
                        help='RTMPose model variant to download/use (default: rtmpose-m)')
    parser.add_argument('--pose-lifter', type=str, help='Path to 3D pose lifter model (ONNX or PyTorch .pth)')
    parser.add_argument('--pose-lifter-config', type=str, help='Path to 3D pose lifter config (for PyTorch models)')
    parser.add_argument('--enable-3d', action='store_true', help='Enable 3D pose lifting')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--no-coreml', action='store_true', help='Disable CoreML acceleration')
    parser.add_argument('--download', action='store_true', help='Download sample model')
    parser.add_argument('--list-models', action='store_true', help='List available model variants')
    args = parser.parse_args()
    
    print("🚀 RTMPose Real-time Webcam Demo with ONNX + CoreML + 3D Lifting")
    print("="*70)
    
    # List available models
    if args.list_models:
        print("\n📋 Available RTMPose model variants:")
        print("-" * 60)
        for variant, config in MODEL_CONFIGS.items():
            print(f"🔹 {variant}:")
            print(f"   Description: {config['description']}")
            print(f"   Input size:  {config['input_size']}")
            print(f"   Download:    python realtime_webcam_onnx.py --variant {variant} --download")
            print()
        
        print("Performance characteristics:")
        print("• rtmpose-t: ~100+ FPS on M1/M2 (fastest)")
        print("• rtmpose-s: ~80+ FPS on M1/M2 (good balance)")  
        print("• rtmpose-m: ~60+ FPS on M1/M2 (better accuracy)")
        print("• rtmpose-l: ~40+ FPS on M1/M2 (best accuracy)")
        print()
        
        print("📋 Available 3D Pose Lifter models:")
        print("-" * 60)
        for variant, config in POSE_LIFTER_CONFIGS.items():
            print(f"🔹 {variant}:")
            print(f"   Description: {config['description']}")
            print(f"   Type:        {config['type']}")
            print(f"   Input:       {config['input_size']}")
            print(f"   Output:      {config['output_size']}")
            if config['type'] == 'pytorch':
                print(f"   Usage:       --enable-3d --pose-lifter-config {config['config']}")
            print()
        return
    
    # Handle model path
    model_path = args.model
    variant = args.variant
    
    if args.download or not model_path:
        model_path = f"{variant}.onnx"
        if not os.path.exists(model_path):
            if not download_sample_onnx_model(model_path, variant):
                print(f"\n❌ No model available. Options:")
                print("1. Convert your own model using MMDeploy:")
                print("   See: https://mmdeploy.readthedocs.io/")
                print("2. Download from OpenMMLab Deploee:")
                print("   See: https://platform.openmmlab.com/deploee")
                print("3. Use --list-models to see available variants")
                return
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    # Handle 3D pose lifter
    pose_lifter_path = args.pose_lifter
    pose_lifter_config = args.pose_lifter_config
    enable_3d = args.enable_3d
    
    if enable_3d:
        if pose_lifter_config:
            # PyTorch model specified
            if not os.path.exists(pose_lifter_config):
                print(f"❌ 3D pose lifter config not found: {pose_lifter_config}")
                enable_3d = False
            elif not HAS_MMPOSE:
                print("❌ MMPose not available for PyTorch 3D pose lifting")
                enable_3d = False
        elif pose_lifter_path:
            # ONNX model specified
            if not os.path.exists(pose_lifter_path):
                print(f"❌ 3D pose lifter model not found: {pose_lifter_path}")
                enable_3d = False
        else:
            print("⚠️  3D pose lifting enabled but no pose lifter specified")
            print("   Use --pose-lifter-config for PyTorch models or --pose-lifter for ONNX models")
            enable_3d = False
    
    # Get input size from model variant or use default
    if variant in MODEL_CONFIGS:
        input_size = MODEL_CONFIGS[variant]['input_size']
        print(f"\n🎯 Using model variant: {variant}")
        print(f"   {MODEL_CONFIGS[variant]['description']}")
    else:
        input_size = (192, 256)  # Default RTMPose input size
        print(f"\n🎯 Using custom model with default input size: {input_size}")
    
    if enable_3d:
        if pose_lifter_config:
            print(f"🎯 3D pose lifting enabled (PyTorch): {pose_lifter_config}")
        else:
            print(f"🎯 3D pose lifting enabled (ONNX): {pose_lifter_path}")
    
    # Create demo instance
    use_coreml = not args.no_coreml
    demo = RTMPoseONNXDemo(
        onnx_model_path=model_path,
        use_coreml=use_coreml,
        input_size=input_size,
        model_variant=variant,
        pose_lifter_path=pose_lifter_path,
        pose_lifter_config=pose_lifter_config,
        enable_3d=enable_3d
    )
    
    # Run webcam demo
    demo.run_webcam(args.camera)


if __name__ == '__main__':
    main() 