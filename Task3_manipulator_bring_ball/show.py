"""
show.py: “机械手平面抓球并跨障送至目标位置“任务 — 训练结果(`outputs/`)演示与录制脚本
    核心功能：
    1. 基础功能：
        1. 实时渲染演示训练结果(`outputs/`)：（默认）
        2. 录制视频：通过 `--mode video` 参数，支持录制视频并保存到对应实验目录的 `videos/` 文件夹中；
    2. 模型选择：
        - 示例: `--exp_name "v2.1_exp-05_PPO_r5e" --choose_model "latest" --match_id 2002944`
        - `--exp_name` 参数：指定了实验目录名称（必须手动传入）
        - `--choose_model` 参数：选择展示对应实验目录下 best、latest、stages 三种结果；
        - `--match_id` 参数：可指定文件名中的特定标识（如步数 '2002944'）来筛选文件；如果不指定，则默认选择 reward 最高的模型；
    3. 泛化性测试：
        step1: 获取 train 阶段环境参数: mjpython Task3_manipulator_bring_ball/show.py --help
        step2: 测试 train 时没见过的 env 初始状态
        演示示例: mjpython Task3_manipulator_bring_ball/show.py --mode human --wall 0.02 --ball 0.30 0.03 --target -0.25 0.4 --exp_name "v2.1_exp-05_PPO_r5e"
"""
import os
import re
import glob
import time
import argparse
import numpy as np
import imageio
# 屏蔽一些不必要的警告
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import PlanarBringBallEnv
from config import TrainConfig

def get_args():
    description = """
    “机械手平面抓球并跨障送至目标位置“任务 — 训练结果(outputs/)演示与录制脚本
    --------------------------------------------------
    泛化性测试:
    参考: train 阶段环境参数初始化 (世界坐标):
        1. 墙体 (Wall):
        - 位置: 固定在 x = 0.2
        - 高度: 0.00 ~ 0.30 (wall_height, defined in config.py)
        2. 球 (Ball) 初始范围:
        - x (随机): [0.25, 0.35] (位于墙右侧)
        - z: 约为 0.023, 可取0.03 (略高于地面，防止穿透)
        3. 目标 (Target) 初始范围:
        - tx (随机): [-0.3, -0.2] (位于墙左侧)
        - tz (随机): [0.05, 0.50] (悬浮或贴地)
    通过调整 --wall, --ball, --target 参数，可以测试模型在训练时未见过的环境初始状态下的表现，验证泛化能力。
    --------------------------------------------------
    调用示例: 
    # 获取帮助信息
    mjpython Task3_manipulator_bring_ball/show.py --help
    # 最简化示例
    mjpython Task3_manipulator_bring_ball/show.py --exp_name "v3.1_exp-01_PPO_r2"
    mjpython Task3_manipulator_bring_ball/show.py --exp_name "v2.1_exp-05_PPO_r5e" --mode video 
    # 完整指令示例（演示）
    mjpython Task3_manipulator_bring_ball/show.py --wall 0.00 --ball 0.30 0.03 --target -0.25 0.4 --exp_name "v3.1_exp-01_PPO_r2" --choose_model "latest" --match_id 20316
    # 完整指令示例（录制） 
    mjpython Task3_manipulator_bring_ball/show.py --wall 0.30 --ball 0.30 0.03 --target -0.25 0.4 --exp_name "v2.1_exp-05_PPO_r5e" --choose_model "latest" --match_id 2002944 --mode video -fps 100
    """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # 基础模式
    parser.add_argument("--mode", type=str, default="human", choices=["human", "video"], 
                        help="运行模式: human (实时渲染) 或 video (录制)")
    
    # 环境控制参数
    parser.add_argument("--wall", type=float, default=0.02, 
                        help="设置墙的高度 (课程学习)")
    parser.add_argument("--ball", type=float, nargs=2, metavar=('X', 'Z'), 
                        help="手动指定球的初始位置; (例如: --ball 0.30 0.03)")
    parser.add_argument("--target", type=float, nargs=2, metavar=('X', 'Z'), 
                        help="手动指定目标的位置; (例如: --target -0.25 0.4)")
    
    # 运行配置
    parser.add_argument("--steps", type=int, default=1000, 
                        help="运行的总步数")
    parser.add_argument("--exp_name", type=str, 
                        help="(outputs 文件夹中）实验目录名称")
    parser.add_argument("--choose_model", type=str, default="latest", choices=["best", "latest", "stages"],
                        help="default='latest', 可选: best; latest; stages, 从outputs/对应目录加载模型和统计数据")
    parser.add_argument("--match_id", type=str, default=None,
                        help="default=None, 默认选择 reward 最高的；可选: 指定文件名中的特定标识（如步数 '2002944'）来筛选文件")
    parser.add_argument("--fps", type=int, default=100, 
                        help="渲染/视频帧率，仿真环境的默认步长为 0.01s, 理论上最大帧率为 100fps; 可以适当调整以平衡预览速度和视频流畅度")
    
    return parser.parse_args()

def main():
    args = get_args()
    cfg = TrainConfig()
    log_dir = os.path.join(cfg.task_dir, "outputs", args.exp_name) 
    choose_model_dir = os.path.join(log_dir, args.choose_model)
    video_folder = os.path.join(log_dir, "videos")

    # -------- 选择 .zip 与 .pkl 文件 --------
    zips = glob.glob(os.path.join(choose_model_dir, "*.zip"))
    pkls = glob.glob(os.path.join(choose_model_dir, "*.pkl"))

    # ⚠️：非 best 文件夹文件，文件名必须包含 'reward-' 或 match_id
    # 情况一：best 文件夹，直接取唯一一个
    if args.choose_model == "best":
        model_path = zips[0]
        stats_path = pkls[0]
    else:
        # 情况二：根据 match_id 按标识符筛选
        if args.match_id:
            target_zips = [f for f in zips if args.match_id in os.path.basename(f)]
            target_pkls = [f for f in pkls if args.match_id in os.path.basename(f)]
            model_path = target_zips[0] if target_zips else zips[0]
            stats_path = target_pkls[0] if target_pkls else pkls[0]
        else:
            # 情况三：默认选择最大 reward 的模型：
            # 按 reward 排序，提取 'reward-' 后的浮点数，从大到小排；这里的正则兼容了负号和浮点数
            zips.sort(key=lambda x: float(re.findall(r"reward-?(-?\d+\.?\d*)", x)[0] if "reward" in x else -1e9), reverse=True)
            pkls.sort(key=lambda x: float(re.findall(r"reward-?(-?\d+\.?\d*)", x)[0] if "reward" in x else -1e9), reverse=True)
            model_path = zips[0]
            stats_path = pkls[0]

    # 打印结果检查
    print(f"找到的统计文件: {stats_path}")
    print(f"找到的模型文件: {model_path}")

    if not stats_path or not model_path:
        raise FileNotFoundError(f"在 {choose_model_dir} 中未找到必要的 .pkl 或 .zip 文件")

    # -------- main --------
    # 1. 选择渲染模式 
    render_mode = "human" if args.mode == "human" else "rgb_array"

    # 2. 环境配置
    def make_env():
        # 如果 render_mode 是 video 模式，Env 内部会初始化 Renderer
        env = PlanarBringBallEnv(model_path=cfg.xml_path, render_mode=render_mode)
        return env

    venv = DummyVecEnv([make_env])
    env = VecNormalize.load(stats_path, venv)
    env.training = False 
    env.norm_reward = False 

    env.env_method("set_wall_height", args.wall)
    if args.ball or args.target:
        env.env_method("set_init_state", ball_xz=args.ball, target_xz=args.target)

    # 3. 加载模型
    model = PPO.load(model_path, env=env)

    # 4. 开始渲染：模式 = human (实时渲染) 或 video (录制视频)
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

        # 5. 导出视频
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