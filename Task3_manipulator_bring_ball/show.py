import os
import time
import argparse
import numpy as np
import imageio  # 用于高质量视频导出

# 屏蔽一些不必要的警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import PlanarBringBallEnv
from config import TrainConfig

def get_args():
    parser = argparse.ArgumentParser(description="SB3 模型高质量演示与录制脚本")
    # 基础模式
    parser.add_argument("--mode", type=str, default="human", choices=["human", "video"], 
                        help="运行模式: human (实时渲染) 或 video (高质量录制)")
    
    # 环境控制参数
    parser.add_argument("--wall", type=float, default=0.0, help="设置墙的高度 (课程学习参数)")
    parser.add_argument("--ball", type=float, nargs=2, metavar=('X', 'Z'), 
                        help="手动指定球的初始位置 (例如: --ball 0.4 0.2)")
    parser.add_argument("--target", type=float, nargs=2, metavar=('X', 'Z'), 
                        help="手动指定目标的位置 (例如: --target -0.4 0.5)")
    
    # 运行配置
    parser.add_argument("--steps", type=int, default=1000, help="运行的总步数")
    parser.add_argument("--exp_name", type=str, default="exp-00_PPO_debug_test011", help="实验目录名称")
    parser.add_argument("--fps", type=int, default=60, help="视频帧率 (推荐 60 以保持丝滑)")
    
    return parser.parse_args()

def main():
    args = get_args()
    cfg = TrainConfig()

    # 1. 路径配置
    log_dir = os.path.join(cfg.task_dir, "outputs", args.exp_name) 
    stats_path = os.path.join(log_dir, "best_vec_normalize.pkl")
    model_path = os.path.join(log_dir, "best_model", "best_model.zip")
    video_folder = os.path.join(log_dir, "videos")

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return

    # 2. 渲染模式确定
    # 注意：录制视频必须使用 rgb_array
    render_mode = "human" if args.mode == "human" else "rgb_array"

    # 3. 创建环境工厂
    def make_env():
        # 如果是 video 模式，PlanarBringBallEnv 内部会初始化 Renderer
        env = PlanarBringBallEnv(model_path=cfg.xml_path, render_mode=render_mode)
        return env

    venv = DummyVecEnv([make_env])

    # 4. 加载归一化统计数据 (pkl)
    if os.path.exists(stats_path):
        print(f"正在加载归一化统计数据: {stats_path}")
        env = VecNormalize.load(stats_path, venv)
        env.training = False 
        env.norm_reward = False 
    else:
        print("警告: 未找到 pkl 文件")
        env = venv

    # 5. 设置环境状态 (墙高、位置)
    env.env_method("set_wall_height", args.wall)
    if args.ball or args.target:
        env.env_method("set_init_state", ball_xz=args.ball, target_xz=args.target)

    # 6. 加载模型
    print(f"正在加载模型: {model_path}")
    model = PPO.load(model_path, env=env)

    # 7. 运行与收集
    frames = []
    obs = env.reset()
    print(f"\n>>> 开始运行 [模式: {args.mode}] [墙高: {args.wall}]")
    
    try:
        for i in range(args.steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            if args.mode == "human":
                env.render()
                time.sleep(0.01) # 稍微减慢预览速度
            else:
                # 录制模式：手动提取帧
                # venv.render() 在 rgb_array 模式下返回的是包含 1 个数组的 list
                frame = env.render() 
                if isinstance(frame, list):
                    frames.append(frame[0])
                else:
                    frames.append(frame)
            
            if dones[0]:
                print(f"Episode 结束 (Step {i})")
                if args.mode == "video": 
                    break # 录制模式通常只录一个完整的 Episode
                obs = env.reset()

        # 8. 高质量导出视频
        if args.mode == "video" and len(frames) > 0:
            os.makedirs(video_folder, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            video_path = os.path.join(video_folder, f"high_quality_wall_{args.wall}_{timestamp}.mp4")
            
            print(f"正在以超高码率合成视频...")
            imageio.mimsave(
                video_path, 
                frames, 
                fps=args.fps, 
                codec='libx264', # 指定使用 H.264 编码
                pixelformat='yuv420p', # 标准像素格式
                # CRF 10 几乎是肉眼无损的级别，preset veryslow 让编码更细腻
                output_params=['-crf', '10', '-preset', 'veryslow'] 
            )
            print(f"✅ 视频已保存至: {video_path}")


    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        env.close()
        print("环境关闭")

if __name__ == "__main__":
    main()