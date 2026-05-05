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
    参考: train 阶段环境参数初始化 (世界坐标,单位统一):
        1. 墙体 (Wall):
        - 位置: 固定在 x = 0.200
        - 高度: {0.000, 0.050, 0.100, 0.250} (wall_height, defined in config.py)
        2. 球 (Ball) 初始范围: 
        - x (随机): [0.250, 0.350] (位于墙右侧)
        - z: 约为 0.032, 可取 0.032 (球的半径为 0.022,略高于地面0.010, 防止穿透)
        3. 目标 (Target) 初始范围:
        - tx (随机): [-0.300, -0.200] (位于墙左侧)
        - tz (随机): [0.050, 0.500] (悬浮或贴地)
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
    """计算 tolerance 奖励。"""
    return tolerance(x, bounds=bounds, margin=margin, sigmoid=sigmoid, value_at_margin=value_at_margin)


class PlanarBringBallEnv(gym.Env):
    """自定义: 具有课程学习功能的 "平面机械臂跨障抓取" MuJoCo 强化学习环境

    该环境模拟了在垂直平面内移动的 一个 4 自由度的机械臂以及 一个 1 自由度的抓取器，
    任务是抓取球体并跨越红色障碍墙，最终将其送达目标点。
    该环境支持动态调整墙高以实现课程学习 (Curriculum Learning)。
    对应“机械手平面抓球并跨障送至目标位置“任务
    
    Attributes:
        model: MuJoCo 的模型实例，对应 "manipulator_bring_ball.xml"
        data: MuJoCo 的数据实例。
        max_steps (int): 每个 episode 的最大步数。
        current_step (int): 当前 episode 已执行的步数。
        action_space (gym.spaces.Box): 5维连续动作空间 (4个臂关节电机 + 1个抓取驱动器),[-1, 1]。
        observation_space (gym.spaces.Box): 15维连续观察空间。
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

        self.cfg = cfg
        self.max_steps = self.cfg.episode_max_steps  # 单个 episode 最大步数限制

        # 定义仿真时间
        self._init_time()

        # 和 xml 交互; 获取 xml 模型元素的 ID（缓存，避免在 step 中频繁查询字符串）
        self._init_ids()

        # Action Space: 
        # 4个机械臂关节 + 1个抓取指令，范围 [-1, 1]
        # 定义：shape == xml 中的 actuator 数量和顺序（root, shoulder, elbow, wrist, grasp） == PPO 的 action_dim = 5
        # 定义：动作一般限制在 [-1, 1]
        # 交互：PPO-Policy Network 输出 action -> step()接收，将 action 写入 self.data.ctrl -> 物理引擎根据 ctrl 计算力、摩擦、碰撞等 -> 更新 self.data 中的位置(xpos)和速度(qvel)等状态变量
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        # Observation Space: 
        # 定义：shape == _get_obs 定义 == PPO 的 input
        # 定义：由 _get_obs 定义
        # 此处只是对外部算法(如 PPO)声明规则(如数据范围), 但不填充数据
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(38,), dtype=np.float32)

        # 初始化，状态变量
        self.current_step = 0
        self.wall_height = 0.0

        # ----- 仅用于 show.py 渲染和泛化性测试，和 train 无关 -----
        # 在 show.py 中，选择固定位置参数
        self.fixed_ball_xz = None
        self.fixed_target_xz = None

        # 渲染器初始化
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model)

    def _init_time(self):
        """初始化仿真时间相关变量。
        """
        self.ctrl_dt = self.cfg.ctrl_dt             # control dt（机器人）
        self.sim_dt = self.cfg.sim_dt               # physics dt（修改 xml）
        self.sim_substeps = self.cfg.sim_substeps

        # 定义单步计算
        self.model.opt.timestep = self.cfg.sim_dt

        # 需确保 sim_dt 能被 ctrl_dt 整除，否则报错
        if abs(self.ctrl_dt / self.sim_dt - self.sim_substeps) > 1e-8 :
            raise ValueError("ctrl_dt must be divisible by sim_dt")


    def _init_ids(self):
        """获取 xml 模型元素的 ID，并缓存，避免在 step 中频繁查询字符串
        
        调用时机：内部函数，必须在环境初始化时调用
        和 xml 交互：传入 self.model (MjModel)—修改物理世界的静态参数
            1. Geom ID: 几何外观，如墙高
            2. SITE ID: 标记点世界坐标,计算目标距离；data.site_xpos[site_id]，调用 site（标记点）的世界坐标
            3. Body ID: 质量、惯性、质心世界坐标；data.body_xpos[body_id]调用 body 的质心世界坐标
            4. Joint ID: 关节位置、速度等状态信息；通过 self.model.jnt_range[joint_id] 获取关节的动作范围
        """
        # Geom ID: 几何外观，如墙高
        self.wall_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "curriculum_wall")

        # SITE ID: 标记点世界坐标,计算目标距离；
        # data.site_xpos[site_id]，调用 site（标记点）的世界坐标
        self.palm_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "palm_touch")  # box 类型标记；判断是否贴掌  # 相对于 hand z 轴向前.043
        self.grasp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp")  # 判断是否“包住”  # 相对于 hand z 轴向前.065
        self.pinch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")  # 对齐物体  # 相对于 hand z 轴向前.090
        self.fingertip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fingertip_touch")  # 食指尖标记
        self.thumbtip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "thumbtip_touch")  # 拇指尖标记

        # Body ID: 质量、惯性、质心世界坐标；
        # data.body_xpos[body_id]调用 body 的质心世界坐标
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        
        # mocap ID
        # target_body_id   = 观测目标位置（用于 reward）
        # target_mocap_id  = 控制目标位置（用于 reset / random goal)
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
        self.target_mocap_id = self.model.body_mocapid[self.target_body_id]

        # --- 用于 observation space ---
        # 机械臂 8 个关节的 qpos 和 qvel 索引（按顺序；确保与 xml 名称一致）
        self.obs_joint_names = [
            "arm_root", "arm_shoulder", "arm_elbow", "arm_wrist",
            "finger", "fingertip", "thumb", "thumbtip"]
        self.joint_qpos_indices = [self.model.joint(name).qposadr[0] for name in self.obs_joint_names]
        self.joint_qvel_indices = [self.model.joint(name).dofadr[0] for name in self.obs_joint_names]
        
        # 5 个触觉传感器的地址（按顺序；确保与 xml 名称一致）
        self.sensor_names = ["palm_touch", "finger_touch", "thumb_touch", "fingertip_touch", "thumbtip_touch"]
        self.sensor_adrs = [self.model.sensor(name).adr[0] for name in self.sensor_names]

        # --- 用于 reward ---
        # 提取前4个主关节["arm_root", "arm_shoulder", "arm_elbow", "arm_wrist"]的索引用于 reward 中的停顿逻辑
        self.arm_qpos_indices = self.joint_qpos_indices[:4]

        # 手部所有 Geom ID (用于碰撞检测)
        hand_geom_names = ["hand", "palm1", "palm2", "thumb1", "thumb2", "thumbtip1", "thumbtip2", 
                           "finger1", "finger2", "fingertip1", "fingertip2"]
        self.hand_geom_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in hand_geom_names]
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
        
        # 给 reward 使用的具体传感器变量
        self.sensor_adr_palm = self.model.sensor("palm_touch").adr[0]
        self.sensor_adr_fingertip = self.model.sensor("fingertip_touch").adr[0]
        self.sensor_adr_thumbtip = self.model.sensor("thumbtip_touch").adr[0]


    def set_wall_height(self, h: float):
        """动态修改障碍墙的高度。用于课程学习调度。

        调用时机：在 train.py 中通过 env.env_method("set_wall_height", stage['wall_height'])调用
        和 xml 交互：传入 self.model (MjModel)—修改物理世界的静态参数
            self.model.geom_size: 修改墙的尺寸（高度的一半，因为 size 定义为半尺寸）
            self.model.geom_pos: 修改墙的位置（z 轴位置为高度的一半，使其底部贴地）

        Args:
            h (float): 新的墙体高度（米）。
        """
        self.wall_height = h
        self.model.geom_size[self.wall_geom_id][2] = h / 2.0
        self.model.geom_pos[self.wall_geom_id][2] = h / 2.0

    def _get_obs(self):
        """从仿真器中提取当前状态并打包为观测向量 (符合指定逻辑并移除Y轴)。
        定义: self.observation_space
        修改环境的实时动态参数, 传入 self.data (MjData)
        """
        # 1. 关节角度 (8个关节的 sin/cos 交替排列) -> 16维
        qpos = self.data.qpos[self.joint_qpos_indices]
        joint_sin = np.sin(qpos)
        joint_cos = np.cos(qpos)
        # 将 sin 和 cos 交替堆叠并打平，结果形如: [sin1, cos1, sin2, cos2, ...]
        arm_pos = np.stack([joint_sin, joint_cos], axis=-1).flatten() 

        # 2. 关节角速度 (8个关节) -> 8维
        arm_vel = self.data.qvel[self.joint_qvel_indices]

        # 3. 触觉传感器 log(1 + touch) -> 5维
        touch_raw = np.array([self.data.sensordata[adr] for adr in self.sensor_adrs])
        # 使用 np.clip 避免物理引擎抖动产生极小负值导致 log1p 报错
        touch_raw = np.clip(touch_raw, 0.0, None)
        touch_log = np.log1p(touch_raw)

        # 4. 空间位置数据 (仅提取 x 和 z 轴, 索引 [0, 2]) -> 6维
        hand_pos = self.data.site_xpos[self.grasp_id][[0, 2]]  # 提取 X 和 Z 坐标 
        object_pos = self.data.xpos[self.ball_body_id][[0, 2]]
        target_pos = self.data.xpos[self.target_body_id][[0, 2]]

        # 5. 球体相对目标位置 (x, z) -> 2维
        rel_pos = object_pos - target_pos

        # 6. 拼接所有观测项 (加上墙体高度特征)
        obs = np.concatenate([
            arm_pos,       # 16维: 关节位置
            arm_vel,       # 8维: 关节速度
            touch_log,     # 5维: 触觉特征
            hand_pos,      # 2维: 抓取点绝对坐标
            object_pos,    # 2维: 球体绝对坐标
            target_pos,    # 2维: 目标点绝对坐标
            rel_pos,       # 2维: 球体相对目标坐标
            [self.wall_height] # 1维: 动态障碍高度 (为了 Curriculum Learning)
        ]).astype(np.float32)

        return obs # 一维 Numpy 数组 (38,)
    

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
        # --- 记录上一时刻的状态变量 ---
        prev_grasp_pos = self.data.site_xpos[self.grasp_id].copy()
        prev_object_pos = self.data.xpos[self.ball_body_id].copy()
        prev_target_pos = self.data.xpos[self.target_body_id].copy()
        prev_dist_b2t = np.linalg.norm(prev_object_pos[..., [0, 2]] - prev_target_pos[..., [0, 2]], axis=-1)  # mocap-target 在 y 轴有前移（为了显示），故此处只用 x，z 轴


        # --- 1. 执行一步 ctrl_dt: 接收 Policy Network 动作输出 -> 分sim_substeps步，物理引擎模拟计算(mj_step/sim_dt)---
        self.data.ctrl[:] = action  # 将 Policy Network的动作输出写入 self.data.ctrl
        for _ in range(self.sim_substeps):
            mujoco.mj_step(self.model, self.data)   # 按 sim_dt 分步计算，计算更精细，防止穿透

        # --- 2. 提取关键物理信息 ---
        # 1. Observation
        obs = self._get_obs()

        # 2. Positions (位置)
        # Body 类型: data.xpos[body_id]调用 body 的质心世界坐标
        object_pos = self.data.xpos[self.ball_body_id]
        target_pos = self.data.xpos[self.target_body_id]

        # SITE 类型: data.site_xpos[site_id]，调用 site（标记点）的世界坐标
        palm_pos = self.data.site_xpos[self.palm_id]  # 相对于 hand z 轴向前.043
        grasp_pos = self.data.site_xpos[self.grasp_id]  # 相对于 hand z 轴向前.065 # Key！抓取物体判断
        pinch_pos = self.data.site_xpos[self.pinch_id]  # 相对于 hand z 轴向前.090 # 对齐 Orient
        fingertip_pos = self.data.site_xpos[self.fingertip_id]  # 食指尖位置
        thumbtip_pos = self.data.site_xpos[self.thumbtip_id]  # 拇指尖位置

        # 3. Kinematics (运动学/距离)
        dist_palm2b = np.linalg.norm(palm_pos - object_pos, axis=-1)
        dist_grasp2b = np.linalg.norm(grasp_pos - object_pos, axis=-1)
        dist_pinch2b = np.linalg.norm(pinch_pos - object_pos, axis=-1)

        # 两指尖到 ball 的平均距离
        dist_ft2b = np.linalg.norm(fingertip_pos - object_pos, axis=-1)  # 2D 距离 且 取(N, 3)的最后一个维度的范数
        dist_tt2b = np.linalg.norm(thumbtip_pos - object_pos, axis=-1)
        dist_at2b = (dist_ft2b + dist_tt2b) / 2.0

        dist_b2t = np.linalg.norm(object_pos[..., [0, 2]] - target_pos[..., [0, 2]], axis=-1)  # mocap-target 在 y 轴有前移（为了显示），故此处只用 x，z 轴

        # 4. Dynamics (动力学/速度相关)
        # 一个ctrl_dt控制步长内，夹爪移动的距离,反映速度
        actual_displacement = np.linalg.norm(grasp_pos - prev_grasp_pos)
        actual_displacement_b2t = np.linalg.norm(object_pos - prev_object_pos)

        # 5. Logic Checks (bool)
        # 5.1. grasp-触觉信号;至少有两个部位接触才认为可能有抓取意图
        touch_threshold = self.cfg.touch_sensor_threshold
        contacts_count = (
            (self.data.sensordata[self.sensor_adr_palm] > touch_threshold) +
            (self.data.sensordata[self.sensor_adr_fingertip] > touch_threshold) +
            (self.data.sensordata[self.sensor_adr_thumbtip] > touch_threshold)
        )
        touch_ok = contacts_count >= 1

        # 5.2. grasp-碰撞检测
        contact_ok = False
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if (con.geom1 == self.ball_geom_id and con.geom2 in self.hand_geom_ids) or \
               (con.geom2 == self.ball_geom_id and con.geom1 in self.hand_geom_ids):
                contact_ok = True
                break

        # 5.3. grasp-抬起高度
        lift_ok = False
        lift_threshold = self.cfg.lift_height_threshold
        lift_ok = object_pos[2] > lift_threshold

        # --- 3. Rewards 计算 ---
        """
        将任务分为以下几个 Phases:
            Phase 1: 
                目标是 is_grasped (成功抓取)；完成目标后满分计算
                分为 5 部分 reward: R11-15
            Phase 2:
                实现成功抓取 -> 成功稳定送至目标
                分为 3 部分 reward: R21-23

        PS: 
            使用 tolerance
            每一种奖励独立出来。清晰，且可以结合 tensorboard
            奖励总和加权平均，天然归一化
        """
        # -- Phase 1 --
        # 严格抓取判定 (抬起高度 > 阈值 且 有碰撞 且 传感器有压力)
        is_grasped = touch_ok and contact_ok and lift_ok
        grasp_mask = 1.0 if is_grasped else 0.0

        # R11: Reach (接近奖励)
        r_reach = _tolerance(dist_grasp2b, bounds=(0.000, 0.007), margin=0.400, sigmoid="gaussian")
        r_reach = r_reach * (1.0 - grasp_mask) + 1.0 * grasp_mask

        # R12_1: Orient (方向对齐奖励)
        site_xmat = self.data.site_xmat[self.pinch_id].reshape(3, 3)  # site_xmat 是 手的抓取轴朝向的 3x3 旋转矩阵，第3列site_xmat[:, 2] 是 Z 轴方向，得到手部“掌心朝向”的单位向量 hand_forward
        hand_forward = site_xmat[:, 2] 
        unit_to_object = (object_pos - pinch_pos) / (np.linalg.norm(object_pos - pinch_pos) + 1e-6)
        dot_product = np.sum(hand_forward * unit_to_object, axis=-1)  # 手的抓取轴朝向和手对球的朝向的点积 # 若为batch，不建议用 dot

        r_orient = _tolerance(dot_product, bounds=(0.95, 1.0), margin=0.5,sigmoid="gaussian") 
        r_orient = r_orient * (1.0 - grasp_mask) + 1.0 * grasp_mask

        # R12_2：Asymmetry（引入一个“手指对称性”惩罚加强 Orient）
        # 如果两根手指距离球的远近一样，说明球在正中间，差值为 0；如果差值很大，说明偏向一侧。
        asymmetry = np.abs(dist_ft2b - dist_tt2b)
        r_symmetry = _tolerance(asymmetry, bounds=(0.000, 0.002), margin=0.100, sigmoid="gaussian")
        r_symmetry = r_symmetry * (1.0 - grasp_mask) + 1.0 * grasp_mask

        r_orient *= r_symmetry

        # R13: Pause (停顿奖励)
        is_time_to_pause = 1.0 if dist_grasp2b < self.cfg.pause_grasp2b_threshold else 0.0

        r_pause = _tolerance(actual_displacement, bounds=(0.000, 0.001), margin=0.005, sigmoid="gaussian")
        r_pause *= is_time_to_pause
        r_pause = r_pause * (1.0 - grasp_mask) + 1.0 * grasp_mask

        # R14: Close (抓取)
        grasp_action = action[4]  # 抓取器grasp动作 (Action Index 4); <motor name="grasp" joint="finger" gear="10" .../> 对应电机的力矩控制而不是位移控制
       
        #is_time_to_close = (r_reach * r_orient) * float(contact_ok) 
        #r_close *= is_time_to_close  # 注不能有加号！ r_close *= (0.7 * r_reach * float(contact_ok) + 0.3 * r_orient * r_pause)

        r_close_intent = _tolerance(grasp_action, bounds=(0.8, 1.0), margin=1.0, sigmoid="gaussian", value_at_margin=0.01)
        # Phase 1: Soft AND 门引导抓取时机；r_pause包含 is_time_to_pause约束，靠近后才有 close reward
        r_approach_grasp = r_close_intent * (r_reach) * float(contact_ok) #  * r_orient * r_pause
        # Phase 2: 抓稳后的维持约束
        r_sustain_grasp = r_close_intent  # 抓稳后，得分直接取决于你的握力
        # 用 grasp_mask 切换阶段
        r_close = r_approach_grasp * (1.0 - grasp_mask) + r_sustain_grasp * grasp_mask

        # R15: Lift_phase1
        is_closed = touch_ok and contact_ok

        r_lift = _tolerance(object_pos[2], bounds=(0.040, 0.500), margin=0.018,sigmoid="linear")  # 鼓励抬高
        r_lift *= is_closed
        r_lift = r_lift * (1.0 - grasp_mask) + 1.0 * grasp_mask

        # -- Phase 2 --
        # R21: Transport (运输)
        r_transport = _tolerance(dist_b2t, bounds=(0.000, self.cfg.success_dist_threshold), margin=0.700,sigmoid="gaussian")
        r_transport = r_transport * grasp_mask

        # R22: Pause2 (接近目标时的停顿奖励)
        is_time_to_pause2 = 1.0 if dist_b2t < self.cfg.pause_b2t_threshold else 0.0

        r_pause2 = _tolerance(actual_displacement_b2t, bounds=(0.000, 0.005), margin=0.020, sigmoid="gaussian")
        r_pause2 *= is_time_to_pause2
        r_pause2 = r_pause2 * grasp_mask

        # R23: Precision (精确运输)
        r_precision = _tolerance(dist_b2t, bounds=(0.000, self.cfg.success_dist_threshold), margin=0.022, sigmoid="gaussian") 
        r_precision = r_precision * grasp_mask

        # --- R+ ---
        # R+: Progress (Potential-based Progress Reward)
        # r_progress 本质上是一个势能差，随着时间积分它最终会趋近于 0（有进有退）;拆开能防止权重互相淹没
        # 根据 Ng 教授的 Reward Shaping 理论，这种形式的奖励不会改变原有的最优策略，但能提供密集的正交梯度。
        progress = prev_dist_b2t - dist_b2t
        # 限制单步进度的上限，防止模型通过高频抖动刷分
        progress = np.clip(progress, -0.010, 0.010) 
        r_progress = progress * 100.0 * grasp_mask # 放大这个微小的增量到[-1, 1]
        r_progress *= self.cfg.transport_progress_scale

        # --- Penalties ---
        # 拆开能防止权重互相淹没
        # 悬停惩罚: 距离手很近 (r_reach很高)，但是没有触发闭合接触
        is_hovering = (r_reach > 0.7) and not contact_ok 
        penalty_hover = is_hovering * (1.0 - grasp_mask) # Phase1：{0，1}
        penalty_hover *= - self.cfg.hover_penalty_scale

        w_11 = float(self.cfg.reach_weight)
        w_12 = float(self.cfg.orient_weight)
        w_13 = float(self.cfg.pause_weight)
        w_14 = float(self.cfg.close_weight)
        w_15 = float(self.cfg.lift_reward_weight)
        w_21 = float(self.cfg.transport_weight)
        w_22 = float(self.cfg.pause2_weight)
        w_23 = float(self.cfg.precision_weight)

        weight_sum = max(w_11 + w_12 + w_13 + w_14 + w_15 + w_21 + w_22 + w_23, 1e-6)

        # 复合总奖励(并限制在 [0, 1] 范围内)
        # Phase1：Reach + Orient + Pause + Close + Lift_phase1 (Close 特殊；Hover 惩罚)
        # Phase2：Transport + Precision (Progress 引力场)
        reward = (
            (
                w_11 * r_reach + w_12 * r_orient + w_13 * r_pause 
                + w_14 * r_close + w_15 * r_lift
                + w_21 * r_transport + w_22 * r_pause2 + w_23 * r_precision 
            )
            / weight_sum  
            + penalty_hover 
            + r_progress
        )

        # --- 4. 终止与信息记录 ---
        self.current_step += 1

        # 成功判定：球离目标极近
        success = (dist_b2t < self.cfg.success_dist_threshold) and (grasp_mask > 0.5)
        
        terminated = False
        is_success_val = 0.0
        if success:
            # reward += w_success  # 额外的通关大奖
            is_success_val = 1.0
            terminated = True

            # 补发通关大奖！(补偿它因为提前结束而没拿到的剩余步骤的分数)
            # 假设每步平均能拿 0.9 分，直接把剩余步数的潜在分数一次性给它
            remaining_steps = self.max_steps - self.current_step
            reward += (remaining_steps * 0.9) + 50.0  # 额外再加 50 庆祝分
            
        truncated = self.current_step >= self.max_steps

        info = {
            "reward_components": {
                "11_reach": r_reach,
                "12_orient": r_orient,     
                "13_pause": r_pause,
                "14_close": r_close,
                "15_lift": r_lift,
                "21_transport": r_transport,
                "22_pause2": r_pause2,
                "23_precision": r_precision,
                "+_progress": r_progress,
                "-_penalty_hover": penalty_hover,
                "total": reward
            },
            "wall_h": self.wall_height,
            "is_grasped": grasp_mask,
            "dist_b2t": dist_b2t,
            "is_success": is_success_val
        }

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