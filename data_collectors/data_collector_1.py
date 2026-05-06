import os
import numpy as np
import cv2
from datetime import datetime
import json
import h5py
from concurrent.futures import ProcessPoolExecutor, Future
from typing import List, Optional, Dict, Any
from glob import glob

def _write_episode_data(episode_path: str, episode_name: str, 
                       camera_data: dict, agent_pose_data: np.ndarray, 
                       actions_data: np.ndarray, language_instruction: Optional[str] = None, 
                       compression=None, image_format: str = "png", camera_configs: List[Dict] = None):
    """Helper function to write episode data in a separate process
    
    Args:
        episode_path: Path to the individual episode HDF5 file
        episode_name: Name of the episode
        camera_data: Dict of camera name to image data {name: [T, H, W, 3]}
        agent_pose_data: Robot joint angles [T, num_joints]
        actions_data: Robot actions [T, num_joints]
        language_instruction: Language instruction for the task
        compression: Compression method for HDF5 data, None for no compression
        image_format: Format for saving images ('png', 'jpg', etc.)
        camera_configs: List of camera configurations
    """
    
    # Create episode directory for images
    episode_dir = os.path.dirname(episode_path)
    images_dir = os.path.join(episode_dir, f"{episode_name}_images")
    os.makedirs(images_dir, exist_ok=True)
    
    print(f"Writing episode {episode_name} to {episode_path}")
    print(f"Saving images to {images_dir}")
    
    # Store camera metadata and image paths
    camera_metadata = {}
    
    # Save camera images to disk and record metadata
    for camera_name, image_data in camera_data.items():
        camera_images_dir = os.path.join(images_dir, camera_name)
        os.makedirs(camera_images_dir, exist_ok=True)
        
        image_paths = []
        # Save each frame as individual image file
        for t in range(image_data.shape[0]):
            image_filename = f"frame_{t:06d}.{image_format}"
            image_path = os.path.join(camera_images_dir, image_filename)
            
            # Convert BGR to RGB if needed (OpenCV uses BGR)
            image_to_save = image_data[t]
            if len(image_to_save.shape) == 3 and image_to_save.shape[2] == 3:
                # Assume it's RGB, but OpenCV expects BGR for saving
                image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGB2BGR)
            
            # Save image
            success = cv2.imwrite(image_path, image_to_save)
            if not success:
                print(f"Warning: Failed to save image {image_path}")
            
            image_paths.append(image_path)
        
        camera_metadata[camera_name] = {
            'image_paths': image_paths,
            'shape': image_data.shape[1:],  # (H, W, C)
            'dtype': str(image_data.dtype),
            'num_frames': len(image_paths)
        }
    
    # Write HDF5 file with metadata and non-image data
    with h5py.File(episode_path, 'w') as h5_file:
        # Store camera configurations
        if camera_configs is not None:
            config_group = h5_file.create_group("camera_configs")
            for i, config in enumerate(camera_configs):
                cam_config_group = config_group.create_group(f"camera_{i}")
                for key, value in config.items():
                    if isinstance(value, (str, int, float, bool)):
                        cam_config_group.attrs[key] = value
                    elif isinstance(value, (list, tuple)):
                        cam_config_group.create_dataset(key, data=value)
        
        # Store camera metadata
        camera_group = h5_file.create_group("cameras")
        for camera_name, metadata in camera_metadata.items():
            cam_group = camera_group.create_group(camera_name)
            # Store image paths as variable length strings
            paths_ds = cam_group.create_dataset(
                "image_paths", 
                data=metadata['image_paths'],
                dtype=h5py.special_dtype(vlen=str)
            )
            cam_group.create_dataset("shape", data=metadata['shape'])
            cam_group.attrs['dtype'] = metadata['dtype']
            cam_group.attrs['num_frames'] = metadata['num_frames']
        
        # Store pose and action data without compression
        h5_file.create_dataset(
            "agent_pose", 
            data=agent_pose_data, 
            dtype='float32', 
            chunks=True
        )
        h5_file.create_dataset(
            "actions", 
            data=actions_data, 
            dtype='float32', 
            chunks=True
        )
        
        # Store language instruction if provided
        if language_instruction is not None:
            h5_file.create_dataset(
                "language_instruction",
                data=language_instruction,
                dtype=h5py.special_dtype(vlen=str)
            )
        
        # Store episode metadata
        h5_file.attrs['episode_name'] = episode_name
        h5_file.attrs['images_dir'] = images_dir
        h5_file.attrs['num_frames'] = agent_pose_data.shape[0]
        h5_file.attrs['timestamp'] = datetime.now().isoformat()
        h5_file.attrs['num_cameras'] = len(camera_data)
    
    print(f"Finished writing episode {episode_name}")

def _load_episode_images(episode_path: str, camera_name: str = None):
    """Helper function to load images from file system for an episode"""
    with h5py.File(episode_path, 'r') as h5_file:
        images_dir = h5_file.attrs['images_dir']
        camera_group = h5_file['cameras']
        
        images_data = {}
        for cam_name in camera_group.keys():
            if camera_name is not None and cam_name != camera_name:
                continue
                
            cam_group = camera_group[cam_name]
            image_paths = [path.decode('utf-8') if isinstance(path, bytes) else path 
                          for path in cam_group['image_paths'][:]]
            
            # Load images
            images = []
            for img_path in image_paths:
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                else:
                    print(f"Warning: Could not load image {img_path}")
                    # Add zero image as placeholder
                    shape = cam_group['shape'][:]
                    images.append(np.zeros(shape, dtype=np.uint8))
            
            images_data[cam_name] = np.array(images)
        
        return images_data

class DataCollector:
    def __init__(self, camera_configs: List[dict], save_dir="output", 
                 max_episodes=10, max_workers=4, compression=None, 
                 image_format: str = "png", image_quality: int = 95):
        """Initialize the data collector with image file storage
        
        Args:
            camera_configs: List of camera configuration dicts with keys:
                - prim_path: Prim path in the simulation
                - name: Camera name (e.g., "camera_1", "camera_2")
                - translation: [x, y, z] position
                - resolution: [width, height]
                - focal_length: Focal length
                - orientation: [x, y, z, w] quaternion
                - image_type: Type of image ("rgb", "depth", etc.)
            save_dir: Root directory for saving data
            max_episodes: Maximum number of episodes to record
            max_workers: Maximum number of parallel processes
            compression: Compression method for HDF5 data
            image_format: Format for saving images ('png', 'jpg')
            image_quality: Quality for JPEG compression (1-100)
        """
        self.save_dir = save_dir
        self.max_episodes = max_episodes
        self.compression = compression
        self.image_format = image_format.lower()
        self.image_quality = image_quality
        self.camera_configs = camera_configs
        
        # Set OpenCV parameters for image saving
        if self.image_format == "jpg" or self.image_format == "jpeg":
            self.image_params = [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
        else:
            self.image_format = "png"  # default to png
            self.image_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]  # PNG compression level
        
        self.session_dir = os.path.join(save_dir, "dataset")
        self.mate_dir = os.path.join(self.session_dir, "meta")
        self.episode_file_path = os.path.join(self.mate_dir, "episode.jsonl")
        self.episode_count = 0
        self.task_instructions = None
        
        # Create directories
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(self.mate_dir, exist_ok=True)
        
        # Save camera configurations
        self._save_camera_configs()
        
        # Initialize temporary storage - use camera names directly from config
        self.temp_cameras = {}
        for config in camera_configs:
            camera_name = config['name']
            # Handle multiple image types if specified with '+'
            if '+' in config['image_type']:
                types = config['image_type'].split('+')
                for t in types:
                    self.temp_cameras[f"{camera_name}_{t}"] = []
            else:
                self.temp_cameras[camera_name] = []
        
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        
        # Initialize process pool
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.pending_futures: List[Future] = []
    
    def _save_camera_configs(self):
        """Save camera configurations to JSON file"""
        config_path = os.path.join(self.mate_dir, "camera_configs.json")
        with open(config_path, 'w') as f:
            json.dump(self.camera_configs, f, indent=2)
        print(f"Camera configurations saved to {config_path}")
        
    def cache_step(self, camera_images: dict, joint_angles: np.ndarray, language_instruction: Optional[str] = None):
        """Cache each step's data in temporary lists
        
        Args:
            camera_images: Dict of camera name to image data {name: np.ndarray}
                         Camera names should match those in camera_configs
            joint_angles: Robot joint angles
            language_instruction: Language instruction for the task
        """
        if self.task_instructions is None and language_instruction is not None:
            self.task_instructions = language_instruction
        
        # Store camera images
        for camera_name, image in camera_images.items():
            if camera_name in self.temp_cameras:
                self.temp_cameras[camera_name].append(image)
            else:
                print(f"Warning: Camera '{camera_name}' not found in configured cameras")
        
        # Store joint angles
        self.temp_agent_pose.append(joint_angles)
        
        # Store language instruction
        if language_instruction is not None:
            self.temp_language_instruction = language_instruction
        
    def write_cached_data(self, final_joint_positions):
        """Write cached data asynchronously using process pool"""
        if self.episode_count >= self.max_episodes:
            self.close()
            return
            
        # Add the final action
        if len(self.temp_agent_pose) > 0:
            self.temp_actions = self.temp_agent_pose[1:] + [final_joint_positions]
        else:
            self.temp_actions = []
        
        # Convert lists to numpy arrays
        camera_data = {}
        for name, images in self.temp_cameras.items():
            if len(images) > 0:
                camera_data[name] = np.array(images)
            else:
                print(f"Warning: No images collected for camera {name}")
        
        agent_pose_data = np.array(self.temp_agent_pose) if len(self.temp_agent_pose) > 0 else np.array([])
        actions_data = np.array(self.temp_actions) if len(self.temp_actions) > 0 else np.array([])
        
        # Check if we have valid data to save
        if agent_pose_data.size == 0:
            print("Warning: No agent pose data to save")
            self.clear_cache()
            return
        
        # Create individual episode file path
        episode_name = f"episode_{self.episode_count:04d}"
        episode_path = os.path.join(self.session_dir, f"{episode_name}.h5")
        
        # Submit writing task to process pool
        future = self.process_pool.submit(
            _write_episode_data,
            episode_path,
            episode_name,
            camera_data,
            agent_pose_data,
            actions_data,
            self.temp_language_instruction,
            self.compression,
            self.image_format,
            self.camera_configs
        )
        self.pending_futures.append(future)

        # Write metadata
        info = {
            "episode_index": self.episode_count,
            "tasks": [self.task_instructions] if self.task_instructions else [],
            "length": len(self.temp_agent_pose),
            "episode_path": episode_path,
            "images_dir": os.path.join(self.session_dir, f"{episode_name}_images"),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(self.episode_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(info, ensure_ascii=False) + "\n")
        
        # Clear cache
        self.clear_cache()
        
        # Increment episode count
        self.episode_count += 1

    def clear_cache(self):
        """Clear the cached data without writing to disk"""
        for camera_name in self.temp_cameras:
            self.temp_cameras[camera_name] = []
        self.temp_agent_pose = []
        self.temp_actions = []
        self.temp_language_instruction = None
        self.task_instructions = None
        
    def close(self):
        """Close the data collector and wait for all writes to complete"""
        # Wait for all pending writing operations to complete
        for future in self.pending_futures:
            try:
                future.result(timeout=30)  # 30 second timeout
            except Exception as e:
                print(f"Error waiting for write operation: {e}")
        
        # Shutdown process pool
        self.process_pool.shutdown(wait=True)
        
        print(f"Data collection completed. {self.episode_count} episodes saved.")
        
    @staticmethod
    def load_episode(episode_path: str, load_images: bool = True, camera_name: str = None):
        """Load episode data from disk
        
        Args:
            episode_path: Path to episode HDF5 file
            load_images: Whether to load images from file system
            camera_name: Specific camera to load (None for all cameras)
            
        Returns:
            Dictionary containing episode data
        """
        with h5py.File(episode_path, 'r') as h5_file:
            # Load non-image data
            agent_pose = h5_file['agent_pose'][:] if 'agent_pose' in h5_file else None
            actions = h5_file['actions'][:] if 'actions' in h5_file else None
            
            language_instruction = None
            if 'language_instruction' in h5_file:
                language_instruction = h5_file['language_instruction'][()]
                if isinstance(language_instruction, bytes):
                    language_instruction = language_instruction.decode('utf-8')
            
            # Load camera configurations if available
            camera_configs = None
            if 'camera_configs' in h5_file:
                config_group = h5_file['camera_configs']
                camera_configs = []
                for cam_key in config_group.keys():
                    cam_group = config_group[cam_key]
                    config = {}
                    for key in cam_group.attrs:
                        config[key] = cam_group.attrs[key]
                    for key in cam_group:
                        config[key] = cam_group[key][:]
                    camera_configs.append(config)
            
            # Load images if requested
            camera_data = {}
            if load_images and 'cameras' in h5_file:
                camera_data = _load_episode_images(episode_path, camera_name)
            elif 'cameras' in h5_file:
                # Just return metadata
                camera_group = h5_file['cameras']
                for cam_name in camera_group.keys():
                    if camera_name is not None and cam_name != camera_name:
                        continue
                    cam_group = camera_group[cam_name]
                    camera_data[cam_name] = {
                        'image_paths': cam_group['image_paths'][:],
                        'shape': cam_group['shape'][:],
                        'num_frames': cam_group.attrs['num_frames']
                    }
            
            return {
                'camera_data': camera_data,
                'agent_pose': agent_pose,
                'actions': actions,
                'language_instruction': language_instruction,
                'camera_configs': camera_configs,
                'metadata': dict(h5_file.attrs)
            }