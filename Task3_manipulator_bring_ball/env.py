"""
env.py: “机械手平面抓球并跨障送至目标位置“任务 — 自定义 MuJoCo 强化学习环境
    核心功能：
    1. 实验-不同奖励设计+权重；
    2. 实验-课程学习（通过调整墙的高度）。
    4. `reset`方法：
        train 阶段，随机初始化球和 target的位置(x, z); 
        show 阶段，支持随机位置初始化（默认）和 手动定义固定位置
        (已限制随机初始化在墙内；限制随机初始化球在墙的右侧, target 在墙的左侧)
    3. 可视化-通过 show.py 可视化训练结果; 录制视频; 
    --------------------------------------------------
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
    --------------------------------------------------
    PS:
    - 对应 MuJoCo 的模型实例: "Task3_manipulator_bring_ball/xml/manipulator_bring_ball.xml" 
    （可通过 mjpython Task3_manipulator_bring_ball/xml/test_xml.py 展示）

    - 任务原型是: DeepMind Control Suite: Manipulator
    - 原始xml来源(略微修改): https://github.com/Motphys/MotrixLab/blob/main/motrix_envs/src/motrix_envs/basic/manipulator/manipulator_bring_ball.xml
"""
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np

from config import TrainConfig
from tolerance import tolerance


def _tolerance(
    x: np.ndarray,
    *,  # 限制后面的参数必须是 keyword-only
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = 0.1,
) -> np.ndarray:
    return tolerance(x, bounds=bounds, margin=margin, sigmoid=sigmoid, value_at_margin=value_at_margin)

class PlanarBringBallEnv(gym.Env):
    """自定义: 具有课程学习功能的 "平面机械臂跨障抓取" MuJoCo 强化学习环境

    该环境模拟了在垂直平面内移动的 一个 4 自由度的机械臂以及 一个 1 自由度的抓取器，
    任务是抓取球体并跨越红色障碍墙，最终将其送达目标点。
    该环境支持动态调整墙高以实现课程学习 (Curriculum Learning)。
    对应“机械手平面抓球并跨障送至目标位置“任务
    
    Attributes:
        model: MuJoCo 的模型实例，对应 "manipulator_bring_ball.xml"
            提供静态参数。比如你手动修改的墙高 self.wall_height
        data: MuJoCo 的数据实例。
            提供动态状态(关节位置qpos、速度qvel、物体的实时坐标xpos)
        max_steps (int): 每个 episode 的最大步数。
        current_step (int): 当前 episode 已执行的步数。
        action_space (gym.spaces.Box): 5维连续动作空间 (4个臂关节电机 + 1个抓取驱动器),[-1, 1]。
        observation_space (gym.spaces.Box): 15维连续观察空间。
            对外部算法(如 PPO)声明规则(如数据范围), 但不填充数据
        wall_height (float): 当前课程学习中的障碍墙高度。
    """

    # 仅用于 show.py，不影响 train
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, model_path="manipulator_bring_ball.xml", render_mode=None, cfg=TrainConfig()):
        """初始化环境。

        Args:
            model_path: MuJoCo XML 模型的路径。
            render_mode: 渲染模式, 默认 None(训练模式);可选 "human" 或 "rgb_array"。
            cfg: 配置字典。包含奖励权重、最大步数等。
        """
        super().__init__()
        # 导入 xml —> data
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 动作空间: 5个电机 (root, shoulder, elbow, wrist, grasp)，一般限制在 [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        # 观察空间: 15维 (关节pos[4], 关节vel[4], hand_xz[2], ball_xz[2], target_xz[2], wall_h[1])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        # 获取 xml 模型元素的 ID（缓存，避免在 step 中频繁查询字符串）
        self.wall_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "curriculum_wall")
        self.pinch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_ball")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
        self.target_mocap_id = self.model.body_mocapid[self.target_body_id]

        # 导入 config
        self.cfg = cfg
        self.max_steps = self.cfg.episode_max_steps  # 单个 episode 最大步数限制
        self.success_threshold = self.cfg.success_threshold

        self.w_1 = self.cfg.reward_weight_1  # reward 权重
        self.w_2 = self.cfg.reward_weight_2
        self.w_3 = self.cfg.reward_weight_3
        self.w_success = self.cfg.reward_weight_success

        # 状态变量
        self.current_step = 0
        self.wall_height = 0.0
        self.fixed_ball_xz = None
        self.fixed_target_xz = None

        # 渲染器初始化，仅用于 show.py，不影响 train
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model)

    def set_wall_height(self, h: float):
        """动态修改障碍墙的高度。用于课程学习调度。

        修改物理世界的静态参数, 传入 self.model (MjModel)

        Args:
            h (float): 新的墙体高度（米）。
        """
        self.wall_height = h
        self.model.geom_size[self.wall_geom_id][2] = h / 2.0
        self.model.geom_pos[self.wall_geom_id][2] = h / 2.0

    def _get_obs(self):
        """从仿真器中提取当前状态并打包为观测向量。
        
        修改环境的实时动态参数, 传入 self.data (MjData)

        Returns:
            np.ndarray: 包含关节状态、末端位置、球位置、目标位置和墙高的 15 维数组。
            对应15维观察空间: (关节pos[4], 关节vel[4], hand_xz[2], ball_xz[2], target_xz[2], wall_h[1])
        """
        hand_pos = self.data.site_xpos[self.pinch_id][[0, 2]]  # 提取 X 和 Z 坐标
        ball_pos = self.data.xpos[self.ball_id][[0, 2]]
        target_pos = self.data.site_xpos[self.target_id][[0, 2]]
        
        return np.concatenate([
            self.data.qpos[:4],      # 4个主关节位置
            self.data.qvel[:4],      # 4个主关节速度
            hand_pos,                # 手部末端位置 (2D)
            ball_pos,                # 球位置 (2D)
            target_pos,              # 目标位置 (2D)
            [self.wall_height]       # 任务环境参数
        ]).astype(np.float32)

    def step(self, action):
        """执行环境的一步模拟并计算奖励。

        (输入动作 -> 物理引擎模拟(mj_step) -> 获得新状态(obs) -> 计算奖励)
        Step1: 执行一步模拟
        1. 将动作(action)写入 self.data.ctrl, 这就像给电机的控制信号。
        2. 调用 mujoco.mj_step。物理引擎会计算力、摩擦、碰撞, 并更新 self.data 里的所有位置(xpos)和速度(qvel)。
        3. 获取更新后的观察值(obs)。
        Step2: 定义奖励

        Args:
            action: 神经网络输出的长度为 5 的动作数组，取值范围 [-1, 1]。

        Returns:
            tuple: 包含 (obs, reward, terminated, truncated, info)。
        """
        # Step1: 执行一步模拟
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        
        # Step2: 定义奖励
        # 本质上分解任务，随着结果朝好的方向发展，reward 始终保持上升趋势，且梯度稳定，且鼓励在“波动端“朝好的方向发展
        # 一种方式处理负奖励：通过 if 语句，在突破点，用一个正数（即突破奖励） - weight*（负奖励），确保奖励随进度正相关
        # 另一种方式：构造平滑的函数适应这种，exp，tanh
        # 最好那个控制奖励，实现梯度稳定
        # 逐步引导或者添加算法本身的探索性
        # 另一种方式：参考xxx，clip等等

        # 计算奖励组件，提取关键物理量（世界坐标下的 XZ 距离）
        hand_pos = self.data.site_xpos[self.pinch_id]
        ball_pos = self.data.xpos[self.ball_id]
        target_pos = self.data.site_xpos[self.target_id]

        dist_h2b = np.linalg.norm(hand_pos - ball_pos)  # 手到球的距离
        dist_b2t = np.linalg.norm(ball_pos - target_pos) # 球到目标的距离

        # R1: 趋近奖励 (Reach)
        # 解释：进入球半径范围 2.2cm 获得满分 1.0；在 50cm 范围内有平滑引导梯度
        reward_reach = _tolerance(dist_h2b, bounds=(0, 0.022), margin=0.5, sigmoid='long_tail')

        # R2: 抓取引导奖励 (Grasp)
        # 逻辑：只有当手离球足够近（< 0.03m）时，才奖励“闭合抓取器”的动作
        # action[4] > 0 表示闭合。我们在 [0.8, 1.0] 范围内给满分。
        reward_grasp = 0.0
        if dist_h2b < 0.03:
            reward_grasp = _tolerance(action[4], bounds=(0.8, 1.0), margin=0.5, sigmoid='linear')

        # R3: 带球奖励 (Bring)
        # 逻辑关键：只有当手离球很近（暗示球被控制住）时，才激活此奖励。
        # 否则 Agent 会在还没抓到球时就想去目标点，导致姿态扭曲。
        reward_bring = 0.0
        if dist_h2b < 0.03:
            # 目标距离核心区设为 5cm，缓冲区设为 80cm（覆盖整个工作空间）
            reward_bring = _tolerance(dist_b2t, bounds=(0, 0.05), margin=0.8, sigmoid='long_tail')

        # 最终奖励加权汇总
        # 总奖励理论上限约为 1.0(reach) + 0.5(grasp) + 2.0(bring) = 3.5
        reward = (self.w_1 * reward_reach + 
                  self.w_2 * reward_grasp + 
                  self.w_3 * reward_bring)
        
        """
        # R1(-): 趋近奖励（鼓励末端靠近球）
        reward_reach = - self.w_1 * dist_h2b 
        # R2(+): 引导抓取：如果手离球很近，给一个额外的“鼓励抓取”奖励
        # 可以通过判断两个手指头的位移，或者简单的距离阈值
        reward_grasp = 0
        if dist_h2b < 0.05:
            # 引导 grasp 关节闭合 (假设 action[4] 是抓取)
            reward_grasp = self.w_2 * (1.0 - dist_h2b/0.05)  # 0.5
        # R3(+): 带球奖励：只有当球离开地面，或者球与手距离极近时，才增加 dist_b2t 的权重
        # 否则 Agent 会在还没碰到球时就想去 target，导致姿态扭曲
        reward_bring = 0
        if dist_h2b < 0.03:
            reward_bring = 2.0 - (self.w_3 * dist_b2t)  # 2

        reward = reward_reach + reward_grasp + reward_bring
        """

        """
        # 绕路引导
        if obs[8] < 0.2 and obs[10] > 0.2:
            dist_to_gate = np.linalg.norm(self.data.site_xpos[self.pinch_id] - [0.2, 0, self.wall_height + 0.1])
            reward -= self.w_gate * dist_to_gate
        """

        # 终止逻辑 & R4(+)reward_success
        self.current_step += 1
        terminated = False
        is_success=0

        if dist_b2t < self.success_threshold:
            reward += self.w_success # 给予一笔“终点奖金”
            terminated = True
            is_success = 1.0
            
        truncated = self.current_step >= self.max_steps

        # 组装 info 监控
        info = {
            "is_success": is_success,
            "dist_b2t": dist_b2t,
            "dist_h2b": dist_h2b,
            "r_reach": reward_reach,
            "r_bring": reward_bring
        }

        # 确保返回的是 Python float 而不是 numpy array
        return obs, float(reward), terminated, truncated, info

    def reset(self, seed: int = None, options: dict = None):
        """重置仿真环境到初始状态。

        train 阶段随机初始化球和 target的位置(x, z); 
        show 阶段支持随机位置（默认）和 手动定义固定位置

        Args:
            seed: 随机种子。
            options: 其他重置选项。

        Returns:
            tuple: (initial_observation, info)
        """
        super().reset(seed=seed) # 此时 self.np_random 已根据 seed 初始化
        mujoco.mj_resetData(self.model, self.data)
        self.current_step = 0

        # 获取球半径，防止硬编码
        ball_radius = self.model.geom_size[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball")][0]

        # --- 1. 球的初始位置 ---
        if self.fixed_ball_xz is not None:
            # 假设传入的是世界坐标。公式：qpos = Target_World - Body_Pos + ref
            # 简化后：qpos = Target_World - 0.4 + 0.4 = Target_World
            bx_qpos = self.fixed_ball_xz[0]
            bz_qpos = max(self.fixed_ball_xz[1], ball_radius + 0.01) # 强制高于地面
        else:
            # 墙右侧世界坐标 [0.25, 0.35]
            bx_qpos = self.np_random.uniform(0.25, 0.35)
            # Z 轴设为半径以上一点点，让它自然落地
            bz_qpos = ball_radius + 0.01 
        
        idx_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
        idx_z = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_z")
        
        # 直接写入 qpos
        self.data.qpos[self.model.jnt_qposadr[idx_x]] = bx_qpos
        self.data.qpos[self.model.jnt_qposadr[idx_z]] = bz_qpos

        # --- 2. 目标 (target_ball) 的初始化 ---
        # 现在直接操作 mocap_pos，它代表整个 body 的世界坐标
        if self.fixed_target_xz is not None:
            tx_world, tz_world = self.fixed_target_xz
        else:
            # 我们想让目标在墙左侧：世界坐标 x ∈ [-0.3, 0.1]
            tx_world = self.np_random.uniform(-0.3, -0.2)  # 调整：(-0.3, -0.2)
            tz_world = self.np_random.uniform(0.05, 0.5) # 让它悬浮或贴地

        # 直接设置 Mocap 的世界坐标 (x, y, z)
        # 注意：y 轴建议保持在 0 附近 (0.001 是为了防止和地板重叠闪烁)
        self.data.mocap_pos[self.target_mocap_id] = [tx_world, 0.001, tz_world]
        
        # 3. 同步
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    # ------ 以下方法主要用于 show.py 的可视化演示和录制功能，与训练无关 ------
    def set_init_state(self, ball_xz=None, target_xz=None):
        """手动指定：球和目标的初始位置 (x, z)。

        仅用于 show.py, 和训练无关
        指定的 (x, z)为世界坐标
        需在 reset 之前调用，则 reset 会使用这些值

        Args:
            ball_xz: 球的世界坐标 [x, z]，例如 [0.3, 0.0]
            target_xz: 目标的世界坐标 [x, z]，例如 [-0.2, 0.1]
        """
        self.fixed_ball_xz = ball_xz
        self.fixed_target_xz = target_xz

    def render(self):
        """渲染环境状态，即“刷帧”，否则在 show 时只能看到静止或者看不到画面。

        仅用于 show.py, 和训练无关
        """
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                
                # --- 设置自由视角的初始默认位置（参数通过 xml/test_xml.py 自动获取） ---
                self.viewer.cam.lookat = [0.070, 0.009, 0.492]    # 相机盯着这个点看
                self.viewer.cam.distance = 2.761            # 视角距离焦点的距离（越大画面越小）
                self.viewer.cam.azimuth = 96.373            # 90度表示从正侧面（Y轴方向）看过去
                self.viewer.cam.elevation = -3.769          # 负值表示向下俯视
                # -----------------------------------

            self.viewer.sync()

        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=1440, width=2560)
            self.renderer.update_scene(self.data, camera="fixed")
            return self.renderer.render()

    def close(self):
        """释放资源并关闭渲染器。
        
        仅用于 show.py, 和训练无关
        """
        if self.viewer is not None:
            self.viewer.close()