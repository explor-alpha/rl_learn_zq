"""
show.py: 平面机械手抓球任务—演示与录制脚本
"""
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
    description = """
    平面机械手抓球任务—演示与录制脚本
    --------------------------------------------------
    参考 - Train 阶段位置初始化 (世界坐标参考):

    1. 墙体 (Wall):
    - 位置: 固定在 x = 0.2
    - 高度: 0.00 ~ 0.30 (wall_height, defined in config.py)

    2. 球 (Ball) 初始范围:
    - x (随机): [0.25, 0.35] (位于墙右侧)
    - z: 约为 0.023, 可取0.03 (略高于地面，防止穿透)

    3. 目标 (Target) 初始范围:
    - tx (随机): [-0.3, -0.2] (位于墙左侧)
    - tz (随机): [0.05, 0.50] (悬浮或贴地)
    --------------------------------------------------
    调用示例: 
    mjpython Task3_manipulator_bring_ball/show.py --help
    mjpython Task3_manipulator_bring_ball/show.py --mode human
    mjpython Task3_manipulator_bring_ball/show.py --mode human --wall 0.20 --ball 0.30 0.03 --target -0.25 0.4 --exp_name "exp-00_PPO_debug_test011"
    mjpython Task3_manipulator_bring_ball/show.py --mode video --wall 0.30 --ball 0.30 0.03 --target -0.25 0.4 --steps 1000 --exp_name "exp-00_PPO_debug_test011" --fps 90    
    """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # 基础模式
    parser.add_argument("--mode", type=str, default="human", choices=["human", "video"], 
                        help="运行模式: human (实时渲染) 或 video (录制)")
    
    # 环境控制参数
    parser.add_argument("--wall", type=float, default=0.30, 
                        help="设置墙的高度 (课程学习)")
    parser.add_argument("--ball", type=float, nargs=2, metavar=('X', 'Z'), 
                        help="手动指定球的初始位置; (例如: --ball 0.30 0.03)")
    parser.add_argument("--target", type=float, nargs=2, metavar=('X', 'Z'), 
                        help="手动指定目标的位置; (例如: --target -0.25 0.4)")
    
    # 运行配置
    parser.add_argument("--steps", type=int, default=1000, 
                        help="运行的总步数")
    parser.add_argument("--exp_name", type=str, default="exp-00_PPO_debug_test011", 
                        help="实验目录名称; 例如 exp-00_PPO_debug_test011")
    parser.add_argument("--fps", type=int, default=90, 
                        help="渲染/视频帧率")
    
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

    # 2. 选择渲染模式
    render_mode = "human" if args.mode == "human" else "rgb_array"

    # 3. 环境
    def make_env():
        # 如果 render_mode 是 video 模式，Env 内部会初始化 Renderer
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
            # SB3 的 VecEnv 中，会自动将 terminated 和 truncated 合并成一个布尔值 done（兼容旧版 gym 协议）
            obs, rewards, dones, infos = env.step(action)
            
            if args.mode == "human":
                env.render()
                time.sleep(1.0 / args.fps) # 稍微减慢预览速度；尽量匹配环境的渲染速度 (1/90s)
            else:
                # 录制模式：手动提取帧；调用env.render() ，刷帧
                frame = env.render() 
                frames.append(frame)
            
            if dones[0]:
                print(f"Episode 结束 (Step {i})")
                if args.mode == "video": 
                    break 
                # 录制模式通常只录一个完整的 Episode
                obs = env.reset()

        # 8. 导出视频
        if args.mode == "video" and len(frames) > 0:
            os.makedirs(video_folder, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            video_name = f"eval_wall_{args.wall}_{timestamp}.mp4"
            video_path = os.path.join(video_folder, video_name)
            
            print(f"正在合成视频...")
            imageio.mimsave(
                video_path, 
                frames, 
                fps=args.fps, 
                codec='libx264', 
                pixelformat='yuv420p', 
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