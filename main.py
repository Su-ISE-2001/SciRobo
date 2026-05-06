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
import datetime
import time
import omni
import omni.physx
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.usd

from factories.robot_factory import create_robot
from utils.object_utils import ObjectUtils
from factories.task_factory import create_task
from factories.controller_factory import create_controller

def _annotate_images(camera_images, text: str, color=(0, 0, 255)):
    """在图像左上角添加简单文本标注。"""
    for img in camera_images:
        cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def save_camera_screenshots(camera_images, camera_names, cfg, episode_num, frame_count, suffix="", annotation_text=None, annotation_color=(0, 0, 255)):
    """分别保存每个摄像头的截图，可选标注异常信息。"""
    # 创建episode特定的文件夹
    episode_screenshot_dir = os.path.join(cfg.multi_run.run_dir, "screenshots", f"episode_{episode_num}")
    os.makedirs(episode_screenshot_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    if annotation_text:
        _annotate_images(camera_images, annotation_text, annotation_color)
    
    saved_paths = []
    for i, (camera_img, camera_name) in enumerate(zip(camera_images, camera_names)):
        # 为每个摄像头创建文件名
        filename = f"episode_{episode_num}_frame_{frame_count}_cam_{camera_name}_{timestamp}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".png"
        
        screenshot_path = os.path.join(episode_screenshot_dir, filename)
        success = cv2.imwrite(screenshot_path, camera_img)
        if success:
            saved_paths.append(screenshot_path)
    
    if saved_paths:
        print(f"已保存 {len(saved_paths)} 张截图到 {episode_screenshot_dir}")
    
    return saved_paths

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
    task.reset()

    frame_count = 0
    last_screenshot_time = time.time()
    screenshot_interval = 2  # 1秒保存一次

    persistent_error_suffix = ""

    while simulation_app.is_running():
        world.step(render=True)
        
        if world.is_stopped():
            task_controller.reset_needed = True
            
        if world.is_playing():
            if task_controller.need_reset() or task.need_reset():
                # 在重置前检查是否已达到最大 episode 数
                # 注意：此时 episode_num() 返回的是已完成（已写入）的 episode 数
                # 优先使用 collector.max_episodes，如果不存在则使用顶层的 max_episodes
                max_episodes = getattr(cfg.collector, 'max_episodes', None) or cfg.max_episodes
                if task_controller.episode_num() >= max_episodes:
                    task_controller.close()
                    simulation_app.close()
                    cv2.destroyAllWindows()
                    break
                
                # 移除重置时的截图保存
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                           
                task_controller.reset()
                task.reset()
                frame_count = 0
                last_screenshot_time = time.time()
                # 新 episode 重置异常标记，避免遗留 erro
                persistent_error_suffix = ""

                continue
                
            state = task.step()
            if state is None:
                continue
            
            action, done, is_success = task_controller.step(state)
            if action is not None:
                robot.get_articulation_controller().apply_action(action)
            # 即使 done=True 也要先消费异常并保存一次截图（用于标注 erro 文件名）
            frame_exception = task_controller.consume_frame_exception() if hasattr(task_controller, "consume_frame_exception") else None
            annotation_text = None
            annotation_color = (0, 0, 255)
            exception_suffix = ""
            if frame_exception:
                annotation_text = frame_exception.get("message")
                # "erro" 作为标签仅用于文件名后缀，不叠加到图像上
                if annotation_text == "erro":
                    annotation_text = None
                annotation_color = frame_exception.get("color", (0, 0, 255))
                exception_suffix = frame_exception.get("suffix", "exception")
                # 一旦捕获异常，后续帧命名持续加入 erro；若传入 persist_suffix 则优先使用
                persistent_error_suffix = frame_exception.get("persist_suffix", "erro")
                # 立即触发一次截图，避免短 episode 未到定时截屏点
                last_screenshot_time = 0
            
            if save_video or show_video:
                camera_images = []
                camera_names = []
                
                # 获取所有摄像头图像和名称
                for idx, (camera_key, image_data) in enumerate(state['camera_display'].items()):
                    display_img = cv2.cvtColor(image_data.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    camera_images.append(display_img)
                    
                    # 使用配置文件中的摄像头名称，如果没有则使用默认名称
                    if hasattr(cfg, 'cameras') and idx < len(cfg.cameras):
                        camera_name = cfg.cameras[idx].get('name', f'cam_{idx}')
                    else:
                        camera_name = f'cam_{idx}'
                    camera_names.append(camera_name)
                
                if camera_images:
                    # 生成用于显示的拼接图像
                    if annotation_text:
                        _annotate_images(camera_images, annotation_text, annotation_color)

                    combined_img = np.hstack(camera_images)
                    total_width = 0
                    for idx, img in enumerate(camera_images):
                        label = f"Camera {idx+1} ({cfg.cameras[idx].image_type})"
                        cv2.putText(combined_img, label, (total_width + 2, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
                        total_width += img.shape[1]
                    
                # 定期保存每个摄像头的截图（每秒一次）
                # 正常任务且 episode 已标记失败时跳过截图，避免失败轨迹落盘
                current_time = time.time()
                if current_time - last_screenshot_time >= screenshot_interval:
                    if not (getattr(task_controller, "failure_mode", None) is None and getattr(task_controller, "_episode_failed", False)):
                        suffix_parts = ["periodic"]
                        if persistent_error_suffix:
                            suffix_parts.append(persistent_error_suffix)
                        # Avoid duplicate tags like "..._erro_erro"
                        if exception_suffix and exception_suffix != "erro" and exception_suffix != persistent_error_suffix:
                            suffix_parts.append(exception_suffix)
                        save_camera_screenshots(
                            camera_images, 
                            camera_names, 
                            cfg, 
                            task_controller._episode_num, 
                            frame_count, 
                            "_".join(suffix_parts),
                            annotation_text=annotation_text,
                            annotation_color=annotation_color
                        )
                    last_screenshot_time = current_time
                    
                    # 手动保存截图（按 's' 键）
                    if show_video and cv2.waitKey(1) & 0xFF == ord('s'):
                        suffix_parts = ["manual"]
                        if persistent_error_suffix:
                            suffix_parts.append(persistent_error_suffix)
                        # Avoid duplicate tags like "..._erro_erro"
                        if exception_suffix and exception_suffix != "erro" and exception_suffix != persistent_error_suffix:
                            suffix_parts.append(exception_suffix)
                        save_camera_screenshots(
                            camera_images,
                            camera_names,
                            cfg,
                            task_controller._episode_num,
                            frame_count,
                            "_".join(suffix_parts),
                            annotation_text=annotation_text,
                            annotation_color=annotation_color
                        )
                    
                    if show_video:
                        cv2.imshow('Camera Views', combined_img)
                        cv2.waitKey(1)
                    # if save_video:
                    #     output_dir = os.path.join(cfg.multi_run.run_dir, "video")
                    #     os.makedirs(output_dir, exist_ok=True)
                    #     output_path = os.path.join(output_dir, f"episode_{task_controller._episode_num}.mp4")
                    #     if video_writer is None:
                    #         height, width = combined_img.shape[:2]
                    #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    #         video_writer = cv2.VideoWriter(output_path, fourcc, 60.0, (width, height))
                    #     video_writer.write(combined_img)

        if done:
            # 任务完成/失败：如正常任务失败，则清理本 episode 的截图目录，避免失败帧落盘
            if getattr(task_controller, "failure_mode", None) is None and getattr(task_controller, "_episode_failed", False):
                import shutil
                episode_dir = os.path.join(cfg.multi_run.run_dir, "screenshots", f"episode_{task_controller._episode_num}")
                if os.path.isdir(episode_dir):
                    try:
                        shutil.rmtree(episode_dir)
                        print(f"已删除失败 episode 截图目录: {episode_dir}")
                    except Exception as e:
                        print(f"删除失败 episode 截图目录出错: {e}")
            task.on_task_complete(is_success)
            continue


if __name__ == "__main__":
 main()