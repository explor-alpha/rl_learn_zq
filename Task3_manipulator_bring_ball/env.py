"""
自定义环境 env.py
    对应任务: 平面机械手抓球——DeepMind Control Suite 中的 Manipulator 经典任务
    对应 MuJoCo 的模型实例: "manipulator_bring_ball.xml" 
        原始xml来源(略微修改): https://github.com/Motphys/MotrixLab/blob/main/motrix_envs/src/motrix_envs/basic/manipulator/manipulator_bring_ball.xml
        "manipulator_bring_ball.xml"可通过 test_xml.py 展示

    目前环境支持：
    1. 奖励设计+权重；
    2. 课程学习（通过调整墙的高度）。
    3. 通过 show.py 可视化训练结果; 录制视频; 
    4. reset 逻辑: train 阶段随机初始化球和 target的位置(x, z); show 阶段支持随机位置（默认）和 手动定义固定位置
        (已限制随机初始化在墙内；限制随机初始化球在墙的右侧, target 在墙的左侧)
"""
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
from config import TrainConfig


class PlanarBringBallEnv(gym.Env):
    """一个平面机械臂将球带到目标点的 MuJoCo 强化学习环境。

    该环境模拟了一个具有 5 个执行器的平面机械臂以及一个球。
    任务是控制机械臂抓取（或推动）一个球，并跨越障碍，将其移动到指定的目标位置。

    PS:
        self.data: 提供动态状态(关节位置qpos、速度qvel、物体的实时坐标xpos)
        self.model: 提供静态参数。比如你手动修改的墙高 self.wall_height
        self.observation_space: 对外部算法(如 PPO)声明规则(如数据范围), 但不填充数据
           
    Attributes:
        model: MuJoCo 的模型实例，对应 "manipulator_bring_ball.xml"
        data: MuJoCo 的数据实例。
        max_steps (int): 每个 episode 的最大步数。
        current_step (int): 当前 episode 已执行的步数。
        action_space (gym.spaces.Box): 动作空间, 5个电机的控制信号 [-1, 1]。
        observation_space (gym.spaces.Box): 15维观察空间。
        wall_height (float): 当前课程学习中的障碍墙高度。
    """

    # 仅用于 show.py，不影响 train
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 90}

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
        
        # 获取 xml 模型元素的 ID
        self.wall_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "curriculum_wall")
        self.pinch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_ball")
        
        # 导入 config
        self.cfg = cfg
        self.max_steps = self.cfg.episode_max_steps  # 单个 episode 最大步数限制
        self.success_threshold = self.cfg.success_threshold

        self.w_h2b = self.cfg.reward_weight_hand_to_ball  # reward 权重
        self.w_b2t = self.cfg.reward_weight_ball_to_target
        self.w_gate = self.cfg.reward_weight_gate
        self.w_success = self.cfg.reward_success

        # 状态变量
        self.current_step = 0
        self.wall_height = 0.0
        self.fixed_ball_xz = None
        self.fixed_target_xz = None

        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
        # 获取该 body 对应的 mocap 索引
        self.target_mocap_id = self.model.body_mocapid[self.target_body_id]

        # 渲染器初始化，仅用于 show.py，不影响 train
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model)

    def set_wall_height(self, h):
        """设置环境墙体高度（主要是课程学习接口）。

        修改物理世界的静态参数, 传入 self.model (MjModel)

        Args:
            h: 墙的高度值。
        """
        self.wall_height = h
        self.model.geom_size[self.wall_geom_id][2] = h / 2.0
        self.model.geom_pos[self.wall_geom_id][2] = h / 2.0

    def _get_obs(self):
        """获取当前环境的观察状态。
        
        修改环境的实时动态参数, 传入 self.data (MjData)

        Returns:
            np.ndarray: 包含关节状态、末端位置、球位置、目标位置和墙高的 15 维数组。
            对应15维观察空间: (关节pos[4], 关节vel[4], hand_xz[2], ball_xz[2], target_xz[2], wall_h[1])
        """
        hand = self.data.site_xpos[self.pinch_id][[0, 2]]
        ball = self.data.xpos[self.ball_id][[0, 2]]
        target = self.data.site_xpos[self.target_id][[0, 2]]
        return np.concatenate([
            self.data.qpos[:4], self.data.qvel[:4], 
            hand, ball, target, [self.wall_height]
            ]).astype(np.float32)

    def step(self, action):
        """执行环境的一步模拟并计算奖励。

        Step1: 执行一步模拟
        (输入动作 -> 物理引擎模拟(mj_step) -> 获得新状态(obs) -> 计算奖励)
        1. 将动作(action)写入 self.data.ctrl, 这就像给电机的控制信号。
        2. 调用 mujoco.mj_step。物理引擎会计算力、摩擦、碰撞, 并更新 self.data 里的所有位置(xpos)和速度(qvel)。
        3. 获取更新后的观察值(obs)。

        Step2: 定义奖励

        Args:
            action: 长度为 5 的动作数组，取值范围 [-1, 1]。

        Returns:
            tuple: 包含 (obs, reward, terminated, truncated, info)。
        """
        # Step1: 执行一步模拟
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        
        # Step2: 定义奖励
        # 奖励：靠近球 + 靠近目标；计算欧氏距离：结果是数学纯量（Scalar）
        dist_h2b = np.linalg.norm(self.data.site_xpos[self.pinch_id] - self.data.xpos[self.ball_id])
        dist_b2t = np.linalg.norm(self.data.xpos[self.ball_id] - self.data.site_xpos[self.target_id])
        
        # 组合总奖励：基础惩罚（距离越远奖励越低）
        reward = -(self.w_h2b * dist_h2b) - (self.w_b2t * dist_b2t)
        
        # 绕路引导逻辑权重化
        if obs[8] < 0.2 and obs[10] > 0.2:
            dist_to_gate = np.linalg.norm(self.data.site_xpos[self.pinch_id] - [0.2, 0, self.wall_height + 0.1])
            reward -= self.w_gate * dist_to_gate

        # 成功终止逻辑
        self.current_step += 1
        terminated = False
        if dist_b2t < self.success_threshold:
            reward += self.w_success # 给予一笔“终点奖金”
            terminated = True
            
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def reset(self, seed: int = None, options: dict = None):
        """重置环境到初始状态。

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

    def set_init_state(self, ball_xz=None, target_xz=None):
        """手动指定show阶段 球和目标 的初始位置 (x, z)。

        仅用于 show.py, 和训练无关
        如果在 reset 之前调用，则 reset 会使用这些值
        注意是世界坐标

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