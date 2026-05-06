import os
import argparse
from isaacsim import SimulationApp

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='LabSim Simulation Environment')
    parser.add_argument('--backend', type=str, default='numpy', 
                       choices=['numpy', 'gpu'], 
                       help='Backend choice: numpy (CPU) or gpu')
    parser.add_argument('--headless', action='store_true', 
                       help='Run in headless mode (default is with GUI)')
    parser.add_argument('--no-video', action='store_true', 
                       help='Disable video display and saving')
    parser.add_argument('--config-name', type=str, default='level3_Heat_Liquid',
                       help='Configuration file name (without .yaml extension)')
    parser.add_argument('--save-images', action='store_true',
                    help='保存每个视角的截图（替代/并行于视频）')
    parser.add_argument('--image-interval', type=int, default=24,
                    help='每隔 N 帧保存一次截图（默认 24，每秒都存1）')
    parser.add_argument('--image-format', type=str, default='png',
                    choices=['png', 'jpg', 'jpeg'],
                    help='截图格式（默认 png）')

    return parser.parse_args()

# Get command line arguments
args = parse_args()

# Set up simulation app based on arguments
simulation_config = {"headless": args.headless}
simulation_app = SimulationApp(simulation_config)

import hydra
from omegaconf import OmegaConf
import cv2
import numpy as np

import omni
import omni.physx
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.usd

from factories.robot_factory import create_robot
from utils.object_utils import ObjectUtils
from factories.task_factory import create_task
from factories.controller_factory import create_controller

def main():
    hydra.initialize(config_path="config", job_name=args.config_name)
    cfg = hydra.compose(config_name=args.config_name)
    os.makedirs(cfg.multi_run.run_dir, exist_ok=True)
    OmegaConf.save(cfg, cfg.multi_run.run_dir + "/config.yaml")

    # Set backend based on command line arguments
    if args.backend == 'gpu':
        world = World(stage_units_in_meters=1, device="cpu")
        physx_interface = omni.physx.get_physx_interface()
        physx_interface.overwrite_gpu_setting(1)
    else:
        world = World(stage_units_in_meters=1.0, physics_prim_path="/physicsScene", backend="numpy")
    
    # Override configuration based on command line arguments
    if args.no_video:
        save_video = False
        show_video = False
    else:
        save_video = True
        show_video = True

    robot = create_robot(
        cfg.robot.type,
        position=np.array(cfg.robot.position)
    )
    
    stage = omni.usd.get_context().get_stage()
    add_reference_to_stage(usd_path=os.path.abspath(cfg.usd_path), prim_path="/World")
    
    ObjectUtils.get_instance(stage)
    
    task = create_task(
        cfg.task_type,
        cfg=cfg,
        world=world,
        stage=stage,
        robot=robot,
    )
    
    task_controller = create_controller(
        cfg.controller_type,
        cfg=cfg,
        robot=robot,
    )
    
    video_writer = None
    frame_idx = 0
    task.reset()
    
    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_stopped():
            task_controller.reset_needed = True
            
        if world.is_playing():
            if task_controller.need_reset() or task.need_reset():
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                           
                task_controller.reset()
                frame_idx = 0
                if task_controller.episode_num() >= cfg.max_episodes:
                    task_controller.close()
                    simulation_app.close()
                    cv2.destroyAllWindows()
                    break
                task.reset()
                
                continue
                
            state = task.step()
            if state is None:
                continue
            
            action, done, is_success = task_controller.step(state)
            if action is not None:
                robot.get_articulation_controller().apply_action(action)
            if done:
                task.on_task_complete(is_success)
                continue
            
            if save_video or show_video:
                camera_images = []
                print("[Run dir]:", os.path.abspath(cfg.multi_run.run_dir))
                print("[Images root]:", os.path.abspath(os.path.join(cfg.multi_run.run_dir, "images")))

                # for _, image_data in state['camera_display'].items():
                #     display_img = cv2.cvtColor(image_data.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                #     camera_images.append(display_img)
                
                # if camera_images:
                #     combined_img = np.hstack(camera_images)
                #     total_width = 0
                #     for idx, img in enumerate(camera_images):
                #         label = f"Camera {idx+1} ({cfg.cameras[idx].image_type})"
                #         cv2.putText(combined_img, label, (total_width + 2, 20),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                #         total_width += img.shape[1]
                #     if show_video:
                #         cv2.imshow('Camera Views', combined_img)
                #         cv2.waitKey(1)
                #     if save_video:
                #         output_dir = os.path.join(cfg.multi_run.run_dir, "video")
                #         os.makedirs(output_dir, exist_ok=True)
                #         output_path = os.path.join(output_dir, f"episode_{task_controller._episode_num}.mp4")
                #         if video_writer is None:
                #             height, width = combined_img.shape[:2]
                #             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                #             video_writer = cv2.VideoWriter(output_path, fourcc, 60.0, (width, height))
                #         video_writer.write(combined_img)
                for _, image_data in state['camera_display'].items():
                    display_img = cv2.cvtColor(image_data.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    camera_images.append(display_img)

                # 新增：逐相机保存截图（每相机各自的文件夹）
                if camera_images and args.save_images and (frame_idx % args.image_interval == 0):
                    # 根目录：run_dir/images/episode_xxx/
                    base_dir = os.path.join(
                        cfg.multi_run.run_dir, "images", f"episode_{task_controller._episode_num:03d}"
                    )
                    os.makedirs(base_dir, exist_ok=True)

                    # 逐相机保存
                    for idx, img in enumerate(camera_images):
                        # 每个相机一个子目录：camera_1_rgb 这样的
                        cam_folder = os.path.join(
                            base_dir, f"camera_{idx+1}_{cfg.cameras[idx].image_type}"
                        )
                        os.makedirs(cam_folder, exist_ok=True)

                        # 文件名：frame_000123.png
                        img_path = os.path.join(
                            cam_folder, f"frame_{frame_idx:06d}.{args.image_format}"
                        )

                        # 写盘（jpg/jpeg 时可加质量参数）
                        if args.image_format.lower() in ("jpg", "jpeg"):
                            cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        else:
                            cv2.imwrite(img_path, img)

                # 帧编号自增
                frame_idx += 1


if __name__ == "__main__":
    main()
