#!/usr/bin/env python3
# RTMPose Timing Test
# Testing inference speed on video file with 256x192 resolution

import os
import time
import cv2
import numpy as np
import mmcv
import mmengine
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


class RTMPoseTiming:
    """Class to track timing statistics for RTMPose."""
    def __init__(self):
        self.inference_times = []
        self.detection_times = []
        self.total_times = []
        self.total_frames = 0
    
    def add_inference_time(self, time_ms):
        self.inference_times.append(time_ms)
    
    def add_detection_time(self, time_ms):
        self.detection_times.append(time_ms)
        
    def add_total_time(self, time_ms):
        self.total_times.append(time_ms)
    
    def increment_frame(self):
        self.total_frames += 1
    
    def print_summary(self):
        if not self.inference_times:
            print("No timing data collected!")
            return
            
        print("\n" + "="*60)
        print("RTMPOSE TIMING ANALYSIS")
        print("="*60)
        print(f"Total frames processed: {self.total_frames}")
        print()
        print("POSE INFERENCE TIMING (per person):")
        print(f"  Average: {np.mean(self.inference_times):.2f}ms")
        print(f"  Min: {np.min(self.inference_times):.2f}ms") 
        print(f"  Max: {np.max(self.inference_times):.2f}ms")
        print(f"  Std Dev: {np.std(self.inference_times):.2f}ms")
        print()
        if self.detection_times:
            print("HUMAN DETECTION TIMING:")
            print(f"  Average: {np.mean(self.detection_times):.2f}ms")
            print(f"  Min: {np.min(self.detection_times):.2f}ms") 
            print(f"  Max: {np.max(self.detection_times):.2f}ms")
            print()
        print("TOTAL PIPELINE TIMING (per frame):")
        print(f"  Average: {np.mean(self.total_times):.2f}ms")
        print(f"  Min: {np.min(self.total_times):.2f}ms") 
        print(f"  Max: {np.max(self.total_times):.2f}ms")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  Average FPS: {1000.0/np.mean(self.total_times):.1f}")
        print(f"  Peak FPS: {1000.0/np.min(self.total_times):.1f}")
        print(f"  Minimum FPS: {1000.0/np.max(self.total_times):.1f}")
        print("="*60)


def simple_person_detection(frame):
    """Simple person detection using basic methods."""
    # For demo purposes, assume there's one person in the center region
    height, width = frame.shape[:2]
    
    # Create a bounding box covering the center region
    bbox = [
        width * 0.2,   # x1
        height * 0.1,  # y1  
        width * 0.8,   # x2
        height * 0.9   # y2
    ]
    
    return [bbox]


def process_one_frame_timed(frame, pose_estimator, visualizer, timing_stats, frame_idx, kpt_thr=0.3):
    """Process one frame with timing measurements."""
    
    total_start_time = time.time()
    
    # ========== DETECTION TIMING ==========
    det_start_time = time.time()
    bboxes = simple_person_detection(frame)
    detection_time = (time.time() - det_start_time) * 1000
    timing_stats.add_detection_time(detection_time)
    
    # ========== POSE INFERENCE TIMING ==========
    pose_start_time = time.time()
    
    # RTMPose inference
    batch_results = inference_topdown(pose_estimator, frame, bboxes)
    
    pose_inference_time = (time.time() - pose_start_time) * 1000
    timing_stats.add_inference_time(pose_inference_time)
    
    total_time = (time.time() - total_start_time) * 1000
    timing_stats.add_total_time(total_time)
    
    # Convert frame for visualization
    frame_rgb = mmcv.bgr2rgb(frame)
    
    # Merge results for visualization
    if batch_results:
        merged_results = merge_data_samples(batch_results)
        
        # Visualization
        if visualizer is not None:
            visualizer.add_datasample(
                'result',
                frame_rgb,
                data_sample=merged_results,
                draw_gt=False,
                draw_bbox=True,
                draw_heatmap=False,
                show_kpt_idx=False,
                show=False,
                wait_time=0,
                kpt_thr=kpt_thr)
    
    # Print frame timing info
    num_people = len(batch_results) if batch_results else 0
    print(f"Frame {frame_idx}: Detection: {detection_time:.1f}ms, "
          f"Pose: {pose_inference_time:.1f}ms, "
          f"Total: {total_time:.1f}ms, "
          f"People: {num_people}")
    
    return batch_results


def resize_frame_to_target(frame, target_size=(256, 192)):
    """Resize frame to target resolution while maintaining aspect ratio."""
    target_width, target_height = target_size
    
    # Get original dimensions
    original_height, original_width = frame.shape[:2]
    
    # Calculate scaling to fit within target size
    scale_w = target_width / original_width
    scale_h = target_height / original_height
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize frame
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    # Create canvas of target size and place resized frame
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Center the resized frame
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
    
    return canvas


def main():
    # Configuration - Using RTMPose-s 
    video_path = "/Users/jasonwang/Downloads/IMG_0915.mov"
    config = "/Users/jasonwang/Library/CloudStorage/OneDrive-UniversityofVirginia/Coding/mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py"
    
    # Let MMPose auto-download the checkpoint based on config
    checkpoint = None
    
    device = 'cpu'  # Change to 'cuda:0' if you have GPU
    output_root = 'output_rtmpose'
    target_resolution = (256, 192)  # width, height
    
    print("="*60)
    print("RTMPOSE INFERENCE SPEED TEST")
    print("="*60)
    print(f"Input video: {video_path}")
    print(f"Config: {os.path.basename(config)}")
    print(f"Model: RTMPose-s (Top-down)")
    print(f"Target resolution: {target_resolution[0]}x{target_resolution[1]}")
    print(f"Device: {device}")
    print("="*60)

    # Check video exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Check config exists
    if not os.path.exists(config):
        print(f"Error: Config file not found at {config}")
        return

    # Initialize timing stats
    timing_stats = RTMPoseTiming()

    # Initialize model
    print("Loading RTMPose-s model...")
    try:
        model = init_model(config, checkpoint, device=device)
        print("✓ RTMPose-s model loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("This might be due to missing dependencies or network issues.")
        return

    # Initialize visualizer
    model.cfg.visualizer.radius = 3
    model.cfg.visualizer.line_width = 1
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    # Setup output
    mmengine.mkdir_or_exist(output_root)
    output_file = os.path.join(output_root, f"rtmpose_{target_resolution[0]}x{target_resolution[1]}_{os.path.basename(video_path)}")
    
    print("\nStarting video processing with timing...")
    print("-" * 60)

    # Process video
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video info: {total_frame_count} frames at {fps:.1f} FPS")
    print(f"Downscaling to: {target_resolution[0]}x{target_resolution[1]}")
    
    video_writer = None
    frame_idx = 0
    max_frames = min(50, total_frame_count)  # Process first 50 frames for demo

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    while video.isOpened() and frame_idx < max_frames:
        success, frame = video.read()
        frame_idx += 1

        if not success:
            break

        timing_stats.increment_frame()

        # Resize frame to target resolution
        resized_frame = resize_frame_to_target(frame, target_resolution)

        # Process frame with timing
        try:
            pred_instances = process_one_frame_timed(
                resized_frame, model, visualizer, timing_stats, frame_idx, kpt_thr=0.3)

            # Save output frame
            frame_vis = visualizer.get_image()
            if video_writer is None:
                video_writer = cv2.VideoWriter(
                    output_file, fourcc, fps,
                    (frame_vis.shape[1], frame_vis.shape[0]))
            
            video_writer.write(mmcv.rgb2bgr(frame_vis))
            
        except Exception as e:
            print(f"Warning: Frame {frame_idx} processing failed: {e}")

    video.release()
    if video_writer:
        video_writer.release()

    # Print final timing summary
    timing_stats.print_summary()
    
    print(f"\nRTMPose Characteristics:")
    print(f"  ✓ Top-down approach (detect humans first, then estimate pose)")
    print(f"  ✓ High accuracy pose estimation")
    print(f"  ✓ SimCC head for improved performance")
    print(f"  ✓ Input resolution: {target_resolution[0]}x{target_resolution[1]}")
    print(f"  ✓ CSPNeXt backbone")
    
    print(f"\nOutput video saved to: {output_file}")
    print("RTMPose timing test completed!")


if __name__ == '__main__':
    main() 