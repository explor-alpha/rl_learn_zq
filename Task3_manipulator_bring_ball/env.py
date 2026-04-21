import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

class PlanarBringBallEnv(gym.Env):
    def __init__(self, model_path="manipulator_bring_ball.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs(), {}