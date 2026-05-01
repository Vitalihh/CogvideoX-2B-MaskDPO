"""
Batch video generation script for masked CogVideoX v2v model with multi-GPU support.
Reads a JSON file containing video metadata and mask folder paths, then generates videos.

Usage:
accelerate launch --num_processes 4 maskDPO_infer.py \
    --json_path ./data.json \
    --model_path THUDM/CogVideoX-2b \
    --output_dir ./outputs \
    --video_root_path ./videos \
    --mask_root_path ./masks
"""

import argparse
import json
import os
import sys
import torch
import random
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.maskDPO_pipe import CogVideoXMaskedVideoToVideoPipeline

from diffusers import CogVideoXDDIMScheduler
from diffusers.utils import export_to_video, load_video


class VideoGenerationDataset(Dataset):
    """Dataset for video generation tasks."""
    
    def __init__(self, filtered_samples):
        """
        Args:
            filtered_samples: List of dicts with 'entry' and 'sample_idx'
        """
        self.samples = filtered_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate function that just returns the batch as-is."""
    return batch


def normalize_optional_path(path_value: str, base_dir: Optional[str] = None):
    """Resolve optional path; relative paths are resolved under base_dir when provided."""
    path_str = str(path_value or "").strip()
    if not path_str:
        return None
    if os.path.isabs(path_str) or not base_dir:
        return path_str
    return os.path.join(base_dir, path_str)


def load_mask_tensor_from_folder(mask_folder_path: str, num_frames: int, height: int, width: int):
    """
    Load frame masks from folder and return tensor in shape [1, 1, F, H, W].
    """
    if not mask_folder_path or not os.path.isdir(mask_folder_path):
        raise ValueError(f"mask_folder_path does not exist or is not a directory: {mask_folder_path}")

    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    mask_files = sorted(
        [
            os.path.join(mask_folder_path, name)
            for name in os.listdir(mask_folder_path)
            if os.path.splitext(name)[1].lower() in valid_exts
        ]
    )

    if len(mask_files) == 0:
        raise ValueError(f"No mask images found in folder: {mask_folder_path}")
    if len(mask_files) == 1 and num_frames > 1:
        mask_files = mask_files * num_frames
    elif len(mask_files) < num_frames:
        raise ValueError(
            f"Mask count ({len(mask_files)}) is smaller than video frame count ({num_frames}). "
            f"Please provide at least {num_frames} masks in {mask_folder_path}."
        )
    elif len(mask_files) > num_frames:
        mask_files = mask_files[:num_frames]

    mask_frames = []
    for mask_file in mask_files:
        mask_pil = Image.open(mask_file).convert("L")
        mask_pil = mask_pil.resize((width, height), Image.NEAREST)
        mask_frames.append(np.array(mask_pil).astype(np.float32) / 255.0)

    mask_3d = np.stack(mask_frames, axis=0)
    return torch.from_numpy(mask_3d).unsqueeze(0).unsqueeze(0)


def load_json_data(json_path: str):
    """Load and parse JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_results_json(results_json_path: str):
    """Load existing results JSON file."""
    if os.path.exists(results_json_path):
        try:
            with open(results_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []


def load_all_results_json(output_dir: str):
    """
    Load all results JSON files (both individual GPU files and merged file).
    
    Returns:
        List of all result entries from all JSON files
    """
    all_results = []
    seen_keys = set()  # To avoid duplicates
    
    # Load merged results if exists
    merged_json_path = os.path.join(output_dir, "generation_results.json")
    if os.path.exists(merged_json_path):
        results = load_results_json(merged_json_path)
        for r in results:
            key = (r.get('video_name'), r.get('sample_idx'))
            if key not in seen_keys:
                all_results.append(r)
                seen_keys.add(key)
    
    # Load individual GPU results
    gpu_json_pattern = os.path.join(output_dir, "generation_results_gpu_*.json")
    for gpu_json_path in glob.glob(gpu_json_pattern):
        results = load_results_json(gpu_json_path)
        for r in results:
            key = (r.get('video_name'), r.get('sample_idx'))
            if key not in seen_keys:
                all_results.append(r)
                seen_keys.add(key)
    
    return all_results


def save_results_json(results_json_path: str, results_data: list):
    """Save results to JSON file."""
    # Create directory if not exists
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)


def append_result_to_json(results_json_path: str, result_entry: dict):
    """
    Append a single result entry to JSON file.
    Thread-safe for single process.
    """
    # Create directory if not exists
    os.makedirs(os.path.dirname(results_json_path), exist_ok=True)
    
    # Load existing data
    existing_data = load_results_json(results_json_path)
    
    # Append new entry
    existing_data.append(result_entry)
    
    # Save back
    save_results_json(results_json_path, existing_data)
    
    return len(existing_data)


def is_already_generated(results_data: list, video_name: str, sample_idx: int):
    """Check if a video has already been generated."""
    for entry in results_data:
        if entry.get('video_name') == video_name and entry.get('sample_idx') == sample_idx:
            return True
    return False


def merge_gpu_results(output_dir: str, num_processes: int):
    """
    Merge all GPU-specific JSON files into a single results file.
    
    Args:
        output_dir: Output directory containing GPU-specific JSON files
        num_processes: Number of GPU processes
    """
    all_results = []
    seen_keys = set()
    
    # Collect results from all GPU files
    for gpu_id in range(num_processes):
        gpu_json_path = os.path.join(output_dir, f"generation_results_gpu_{gpu_id}.json")
        if os.path.exists(gpu_json_path):
            results = load_results_json(gpu_json_path)
            # Deduplicate
            unique_results = []
            for r in results:
                key = (r.get('video_name'), r.get('sample_idx'))
                if key not in seen_keys:
                    unique_results.append(r)
                    seen_keys.add(key)
            all_results.extend(unique_results)
            print(f"Loaded {len(results)} results ({len(unique_results)} unique) from GPU {gpu_id}")
    
    # Save merged results
    merged_json_path = os.path.join(output_dir, "generation_results.json")
    save_results_json(merged_json_path, all_results)
    print(f"\nMerged {len(all_results)} total unique results into {merged_json_path}")
    
    return all_results


def generate_single_video(
    pipe,
    prompt: str,
    video_path: str,
    mask_folder_path: str,
    output_paths: dict,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    seed: int = 42,
):
    """
    Generate a single video and save all outputs.
    
    Args:
        output_paths: dict with keys 'video', 'mask', 'latent'

    Returns:
        success (bool), error (str or None), strength (float or None)
    """
    try:
        if not mask_folder_path:
            raise ValueError("mask_folder_path is required for maskDPO_pipe.")
        if not os.path.isdir(mask_folder_path):
            raise ValueError(f"mask folder not found: {mask_folder_path}")

        video = load_video(video_path)[:49]
        video_frames = [img.resize((720, 480), Image.LANCZOS) for img in video]
        if len(video_frames) == 0:
            raise ValueError(f"No frames decoded from input video: {video_path}")

        strength = random.uniform(0.75, 0.95)

        video_generate, video_pt = pipe(
            prompt=prompt,
            video=video_frames,
            mask_folder_path=mask_folder_path,
            strength=strength,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            height=480,
            width=720,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        )

        video_generate = video_generate.frames[0]

        mask_tensor = load_mask_tensor_from_folder(
            mask_folder_path=mask_folder_path,
            num_frames=len(video_frames),
            height=480,
            width=720,
        )
        torch.save(mask_tensor, output_paths["mask"])
        torch.save(video_pt, output_paths["latent"])
        export_to_video(video_generate, output_paths["video"], fps=8)

        return True, None, strength
    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return False, error_msg, None


def batch_generate_videos(
    json_path: str,
    model_path: str,
    output_dir: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    start_idx: int = 0,
    end_idx: int = None,
    num_samples_per_video: int = 1,
    batch_size: int = 1,
    video_name_key: str = "video_name",
    video_root_path: str = None,
    mask_root_path: str = None,
):
    """
    Batch generate videos from JSON data using multiple GPUs.
    
    Parameters:
    - json_path: Path to the JSON file containing video metadata
    - model_path: Path to the pre-trained model
    - output_dir: Directory to save generated videos
    - start_idx: Starting index in the JSON array (for resuming)
    - end_idx: Ending index in the JSON array (None means process all)
    - num_samples_per_video: Number of samples to generate per video entry
    - batch_size: Batch size for DataLoader (usually 1 for video generation)
    - video_name_key: Key in each json entry that stores video file name
    - video_root_path: Root folder for videos. Video is read from video_root_path/video_name
    - mask_root_path: Root folder for masks. Mask folder is read from mask_root_path/<video_stem>
    """
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Only main process creates directories
    if accelerator.is_main_process:
        # Create output directory structure
        video_dir = os.path.join(output_dir, "video")
        mask_dir = os.path.join(output_dir, "mask")
        latent_dir = os.path.join(output_dir, "latents")
        
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(latent_dir, exist_ok=True)
    
    # Wait for directory creation
    accelerator.wait_for_everyone()
    
    # All processes load data
    print(f"[GPU {accelerator.process_index}] Loading JSON data from {json_path}...")
    data = load_json_data(json_path)
    json_dir = os.path.dirname(os.path.abspath(json_path))
    video_root_path = normalize_optional_path(video_root_path, json_dir)
    mask_root_path = normalize_optional_path(mask_root_path, json_dir)
    if not video_root_path:
        raise ValueError("--video_root_path is required.")
    if not mask_root_path:
        raise ValueError("--mask_root_path is required.")
    
    if accelerator.is_main_process:
        print(f"Total entries in JSON: {len(data)}")
    
    # Slice data based on start_idx and end_idx
    if end_idx is None:
        end_idx = len(data)
    data = data[start_idx:end_idx]
    
    if accelerator.is_main_process:
        print(f"Processing entries from {start_idx} to {end_idx} ({len(data)} entries)")
    
    # Load all existing results
    if accelerator.is_main_process:
        print("Loading existing results from all JSON files...")
    
    results_data = load_all_results_json(output_dir)
    
    if accelerator.is_main_process:
        print(f"Loaded {len(results_data)} existing results")
        print(f"Checking files in {output_dir}:")
        # List all JSON files
        all_jsons = glob.glob(os.path.join(output_dir, "*.json"))
        for json_file in all_jsons:
            print(f"  - {os.path.basename(json_file)}")
    
    # GPU-specific results JSON path
    gpu_results_json_path = os.path.join(output_dir, f"generation_results_gpu_{accelerator.process_index}.json")
    print(f"[GPU {accelerator.process_index}] Will save results to: {gpu_results_json_path}")
    
    # Filter out already generated videos
    filtered_data = []
    skipped_missing_video_name = 0
    skipped_missing_video = 0
    skipped_missing_mask = 0
    for entry in data:
        video_name_raw = str(entry.get(video_name_key, "") or "").strip()
        if not video_name_raw:
            skipped_missing_video_name += 1
            continue

        video_path = os.path.join(video_root_path, video_name_raw)
        video_name = video_name_raw
        video_stem = Path(video_name_raw).stem
        mask_folder_path = os.path.join(mask_root_path, video_stem)
        if not os.path.exists(video_path):
            skipped_missing_video += 1
            continue
        if not os.path.isdir(mask_folder_path):
            skipped_missing_mask += 1
            continue

        for sample_idx in range(num_samples_per_video):
            if not is_already_generated(results_data, video_name, sample_idx):
                filtered_data.append({
                    'entry': {
                        **entry,
                        "video_path": video_path,
                        "video_name": video_name_raw,
                        "video_stem": video_stem,
                    },
                    'sample_idx': sample_idx,
                    'mask_folder_path': mask_folder_path,
                })
    
    if accelerator.is_main_process:
        print(f"Skipped entries with missing {video_name_key}: {skipped_missing_video_name}")
        print(f"Skipped entries with missing video file (video_root_path/video_name): {skipped_missing_video}")
        print(f"Skipped entries with missing mask folder (mask_root_path/<video_stem>): {skipped_missing_mask}")
        print(f"After filtering, {len(filtered_data)} samples need to be generated")
    
    # Create dataset and dataloader
    dataset = VideoGenerationDataset(filtered_data)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # Important: use 0 for video generation
    )
    
    # Initialize pipeline on each GPU
    print(f"[GPU {accelerator.process_index}] Loading model from {model_path}...")
    
    pipe = CogVideoXMaskedVideoToVideoPipeline.from_pretrained(
        model_path, 
        torch_dtype=dtype
    )
    
    # Move pipeline to device
    pipe = pipe.to(accelerator.device)
    
    # Set scheduler
    pipe.scheduler = CogVideoXDDIMScheduler.from_config(
        pipe.scheduler.config, 
        timestep_spacing="trailing"
    )
    
    # Enable optimizations
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    print(f"[GPU {accelerator.process_index}] Model loaded successfully!")
    
    # Prepare dataloader with accelerator
    dataloader = accelerator.prepare(dataloader)
    
    if accelerator.is_main_process:
        print(f"DataLoader prepared. Each GPU will process approximately {len(dataloader)} batches")
    
    print(f"[GPU {accelerator.process_index}] Processing {len(dataloader)} batches")
    
    # Create log file for this process
    log_path = os.path.join(output_dir, f"generation_log_gpu_{accelerator.process_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    log_file = open(log_path, 'w', encoding='utf-8')
    log_file.write(f"Batch Video Generation Log - GPU {accelerator.process_index}\n")
    log_file.write(f"Started at: {datetime.now()}\n")
    log_file.write(f"Model: {model_path}\n")
    log_file.write(f"GPU ID: {accelerator.process_index}\n")
    log_file.write(f"Total GPUs: {accelerator.num_processes}\n")
    log_file.write(f"Processing entries: {start_idx} to {end_idx}\n")
    log_file.write(f"Batches to process: {len(dataloader)}\n")
    log_file.write(f"Results JSON: {gpu_results_json_path}\n")
    log_file.write("="*80 + "\n\n")
    log_file.flush()
    
    # Process each batch
    success_count = 0
    fail_count = 0
    
    # Use tqdm for progress tracking (only show on main process)
    if accelerator.is_main_process:
        progress_bar = tqdm(total=len(dataloader), desc=f"GPU {accelerator.process_index}")
    
    for batch_idx, batch in enumerate(dataloader):
        # batch is a list with one item (since batch_size=1)
        for item in batch:
            entry = item['entry']
            sample_idx = item['sample_idx']
            mask_folder_path = item.get('mask_folder_path')
            
            label = entry.get("label", "unknown")
            captions = entry.get("captions", "")
            video_path = entry.get("video_path", "")
            video_file_name = str(entry.get("video_name", "") or "").strip()
            video_stem = str(entry.get("video_stem", "") or "").strip()
            tracker = entry.get("tracker", "")
            
            # Generate output filename
            if not video_file_name:
                video_file_name = Path(video_path).name
            if not video_stem:
                video_stem = Path(video_file_name).stem
            base_filename = f"{video_stem}_sample_{sample_idx}"
            
            # Define output paths
            output_paths = {
                'video': os.path.join(output_dir, "video", f"{base_filename}.mp4"),
                'mask': os.path.join(output_dir, "mask", f"{base_filename}.pt"),
                'latent': os.path.join(output_dir, "latents", f"{base_filename}.pt"),
            }
            
            # Log current processing
            log_msg = f"[GPU {accelerator.process_index}][Batch {batch_idx+1}/{len(dataloader)}] Processing: {video_file_name}, Sample: {sample_idx}, Label: {label}\n"
            log_msg += f"  Input: {video_path}\n"
            log_msg += f"  Mask Folder: {mask_folder_path}\n"
            log_msg += f"  Output Video: {output_paths['video']}\n"
            
            log_file.write(log_msg)
            log_file.flush()
            
            # Check if input video exists
            if not os.path.exists(video_path):
                error_msg = f"  ERROR: Input video not found: {video_path}\n"
                log_file.write(error_msg)
                fail_count += 1
                continue

            if not mask_folder_path or not os.path.isdir(mask_folder_path):
                error_msg = f"  ERROR: Mask folder not found: {mask_folder_path}\n"
                log_file.write(error_msg)
                fail_count += 1
                continue
            
            # Generate video with different seed for each sample
            current_seed = seed + sample_idx + accelerator.process_index * 10000
            success, error, strength = generate_single_video(
                pipe=pipe,
                prompt=captions,
                video_path=video_path,
                mask_folder_path=mask_folder_path,
                output_paths=output_paths,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_videos_per_prompt=num_videos_per_prompt,
                seed=current_seed,
            )
            
            if success:
                success_count += 1
                
                # Create result entry
                result_entry = {
                    "video_name": video_file_name,
                    "video_stem": video_stem,
                    "sample_idx": sample_idx,
                    "label": label,
                    "tracker": tracker,
                    "video_file_name": video_file_name,
                    "prompt": captions,
                    "video_path": output_paths['video'],
                    "mask_path": output_paths['mask'],
                    "latent_path": output_paths['latent'],
                    "source_video": video_path,
                    "mask_folder_path": mask_folder_path,
                    "seed": current_seed,
                    "strength": strength,  # Save the random strength value
                    "gpu_id": accelerator.process_index,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Append to GPU-specific JSON file immediately
                total_entries = append_result_to_json(gpu_results_json_path, result_entry)
                
                log_file.write(f"  SUCCESS: All outputs saved\n")
                log_file.write(f"  Strength: {strength:.4f}\n")
                log_file.write(f"  Result appended to {gpu_results_json_path} (total: {total_entries} entries)\n")
                
                # Print every 10 successes
                if success_count % 10 == 0:
                    print(f"[GPU {accelerator.process_index}] Progress: {success_count} videos generated, saved to {gpu_results_json_path}")
            else:
                fail_count += 1
                log_file.write(f"  FAILED: {error}\n")
            
            log_file.write("\n")
            log_file.flush()
        
        # Update progress bar
        if accelerator.is_main_process:
            progress_bar.update(1)
    
    if accelerator.is_main_process:
        progress_bar.close()
    
    # Write summary for this GPU
    summary = f"\n{'='*80}\n"
    summary += f"GPU {accelerator.process_index} Generation Complete\n"
    summary += f"Finished at: {datetime.now()}\n"
    summary += f"Successful: {success_count}\n"
    summary += f"Failed: {fail_count}\n"
    summary += f"Results saved to: {gpu_results_json_path}\n"
    if (success_count + fail_count) > 0:
        summary += f"Success rate: {success_count/(success_count + fail_count)*100:.2f}%\n"
    
    log_file.write(summary)
    log_file.close()
    print(f"\n[GPU {accelerator.process_index}] Log saved to: {log_path}")
    print(f"[GPU {accelerator.process_index}] Results JSON: {gpu_results_json_path}")
    
    # Verify JSON was created
    if os.path.exists(gpu_results_json_path):
        saved_results = load_results_json(gpu_results_json_path)
        print(f"[GPU {accelerator.process_index}] Verified: {len(saved_results)} results in JSON file")
    else:
        print(f"[GPU {accelerator.process_index}] WARNING: JSON file not found at {gpu_results_json_path}")
    
    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    
    # Gather statistics from all processes
    success_counts = accelerator.gather(torch.tensor([success_count], device=accelerator.device))
    fail_counts = accelerator.gather(torch.tensor([fail_count], device=accelerator.device))
    
    # Merge results and write final summary (only main process)
    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("Merging results from all GPUs...")
        
        # Merge all GPU-specific JSON files
        merged_results = merge_gpu_results(output_dir, accelerator.num_processes)
        
        total_success = success_counts.sum().item()
        total_fail = fail_counts.sum().item()
        
        final_summary = f"\n{'='*80}\n"
        final_summary += f"All GPUs Generation Complete\n"
        final_summary += f"Finished at: {datetime.now()}\n"
        final_summary += f"Total successful: {total_success}\n"
        final_summary += f"Total failed: {total_fail}\n"
        final_summary += f"Total results in merged file: {len(merged_results)}\n"
        if (total_success + total_fail) > 0:
            final_summary += f"Overall success rate: {total_success/(total_success + total_fail)*100:.2f}%\n"
        
        print(final_summary)
        
        # Save final summary
        final_summary_path = os.path.join(output_dir, f"final_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(final_summary_path, 'w') as f:
            f.write(final_summary)
        
        print(f"Final summary saved to: {final_summary_path}")
        print(f"Merged results saved to: {os.path.join(output_dir, 'generation_results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch generate videos from JSON data using masked CogVideoX pipeline with multi-GPU"
    )
    
    # Required arguments
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file containing video metadata")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated videos")
    
    # Optional arguments
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale for generation")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos per prompt")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"],
                        help="Data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index in JSON array")
    parser.add_argument("--end_idx", type=int, default=None, help="Ending index in JSON array (None = all)")
    parser.add_argument("--num_samples_per_video", type=int, default=1, 
                        help="Number of samples to generate per video entry")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for DataLoader")
    parser.add_argument(
        "--video_name_key",
        type=str,
        default="video_name",
        help="Key in each JSON entry that stores video file name",
    )
    parser.add_argument(
        "--video_root_path",
        type=str,
        required=True,
        help="Root folder for videos. Each video is read from video_root_path/video_name",
    )
    parser.add_argument(
        "--mask_root_path",
        type=str,
        required=True,
        help="Root folder for masks. Each mask folder is read from mask_root_path/<video_stem>",
    )
    
    args = parser.parse_args()
    
    # Convert dtype
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    # Run batch generation
    batch_generate_videos(
        json_path=args.json_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        seed=args.seed,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        num_samples_per_video=args.num_samples_per_video,
        batch_size=args.batch_size,
        video_name_key=args.video_name_key,
        video_root_path=args.video_root_path,
        mask_root_path=args.mask_root_path,
    )
