import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np

class PlanarBringBallEnv(gym.Env):
    # 必须添加 metadata，SB3 录像工具会检查这里
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 90}

    def __init__(self, model_path="manipulator_bring_ball.xml", render_mode=None):
        super().__init__()

        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(model_path)



        self.data = mujoco.MjData(self.model)

        # 渲染器初始化
        self.viewer = None
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model)


        self.max_steps = 1000  # 最大步数限制
        self.current_step = 0

        # 动作空间: 5个电机 (root, shoulder, elbow, wrist, grasp)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        
        # 观察空间: 15维 (关节pos/vel, hand, ball, target, wall_h)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        
        # 获取 ID
        self.wall_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "curriculum_wall")
        self.pinch_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_ball")
        
        self.wall_height = 0.0

    def set_wall_height(self, h):
        self.wall_height = h
        self.model.geom_size[self.wall_geom_id][2] = h / 2.0
        self.model.geom_pos[self.wall_geom_id][2] = h / 2.0

    def _get_obs(self):
        hand = self.data.site_xpos[self.pinch_id][[0, 2]]
        ball = self.data.xpos[self.ball_id][[0, 2]]
        target = self.data.site_xpos[self.target_id][[0, 2]]
        return np.concatenate([
            self.data.qpos[:4], self.data.qvel[:4], 
            hand, ball, target, [self.wall_height]
        ]).astype(np.float32)

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        # 简化奖励：靠近球 + 靠近目标
        dist_h2b = np.linalg.norm(self.data.site_xpos[self.pinch_id] - self.data.xpos[self.ball_id])
        dist_b2t = np.linalg.norm(self.data.xpos[self.ball_id] - self.data.site_xpos[self.target_id])
        
        reward = -dist_h2b - dist_b2t * 1.5
        
        # 绕路奖励逻辑
        if obs[8] < 0.2 and obs[10] > 0.2: # 手在左墙，球在右墙
            dist_to_gate = np.linalg.norm(self.data.site_xpos[self.pinch_id] - [0.2, 0, self.wall_height+0.1])
            reward -= dist_to_gate

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def set_init_state(self, ball_xz=None, target_xz=None):
        """
        用于训练后的可视化
        手动指定球和目标的初始化位置 (x, z)
        如果在 reset 之前调用，则 reset 会使用这些值
        """
        self.fixed_ball_xz = ball_xz
        self.fixed_target_xz = target_xz

    # 修改 reset 方法以支持固定位置
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        if hasattr(self, 'fixed_ball_xz') and self.fixed_ball_xz is not None:
            # 分别修改 x 和 z 两个滑动关节
            idx_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_x")
            idx_z = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_z")
            
            self.data.qpos[self.model.jnt_qposadr[idx_x]] = self.fixed_ball_xz[0]
            self.data.qpos[self.model.jnt_qposadr[idx_z]] = self.fixed_ball_xz[1]

        if hasattr(self, 'fixed_target_xz') and self.fixed_target_xz is not None:
            # 修改目标 site 的位置
            self.model.site_pos[self.target_id][0] = self.fixed_target_xz[0]
            self.model.site_pos[self.target_id][2] = self.fixed_target_xz[1]

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    
    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                # 保持 2K 分辨率
                self.renderer = mujoco.Renderer(self.model, height=1440, width=2560)
            
            self.renderer.update_scene(self.data, camera="fixed")
            
            return self.renderer.render()



        
    def close(self):
        """释放资源"""
        if self.viewer is not None:
            self.viewer.close()

    # 注意：在你的 reset 或 step 之后，如果是在可视化，可以手动调一次 self.render()