import mujoco
import numpy as np
import time

class SimplePlanarEnv:
    def __init__(self, model_path="xml/manipulator_bring_ball.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # 核心 ID 获取
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.target_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_ball")
        self.pinch_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.wall_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "curriculum_wall")
        
        # 关节索引 (用于 reset)
        self.arm_joints = ["arm_root", "arm_shoulder", "arm_elbow", "arm_wrist"]
        self.qpos_indices = [self.model.joint(n).qposadr[0] for n in self.arm_joints]
        
        self.wall_height = 0.01 # 初始墙高

    def set_wall_height(self, height):
        """课程学习：动态改墙高"""
        self.wall_height = height
        # 修改几何体高度 (box 的 size[2] 是半高)
        self.model.geom_size[self.wall_geom_id][2] = height / 2.0
        # 修改位置 (确保底部贴地)
        self.model.geom_pos[self.wall_geom_id][2] = height / 2.0

    def get_obs(self):
        """最简观察空间"""
        # 1. 关节位置和速度 (8维)
        # 2. 夹爪、球、目标的 XZ 坐标 (6维)
        # 3. 墙的高度 (1维)
        hand_pos = self.data.site_xpos[self.pinch_site_id][[0, 2]]
        ball_pos = self.data.body_xpos[self.ball_id][[0, 2]]
        target_pos = self.data.site_xpos[self.target_site_id][[0, 2]]
        
        obs = np.concatenate([
            self.data.qpos[self.qpos_indices],
            self.data.qvel[self.qpos_indices],
            hand_pos, ball_pos, target_pos,
            [self.wall_height]
        ])
        return obs.astype(np.float32)

    def compute_reward(self):
        """核心魔改奖励逻辑"""
        hand_pos = self.data.site_xpos[self.pinch_site_id]
        ball_pos = self.data.body_xpos[self.ball_id]
        target_pos = self.data.site_xpos[self.target_site_id]
        
        # 1. 距离计算
        dist_h2b = np.linalg.norm(hand_pos - ball_pos)
        dist_b2t = np.linalg.norm(ball_pos - target_pos)
        
        # 2. 基础奖励：靠近球
        reward = -dist_h2b 
        
        # 3. 抓取与避障逻辑 (课程学习关键)
        # 如果球和手之间有墙 (手在 x<0.2, 球在 x>0.2)
        if hand_pos[0] < 0.2 and ball_pos[0] > 0.2:
            # 引导点：墙上方 10cm
            waypoint = np.array([0.2, 0, self.wall_height + 0.1])
            dist_to_waypoint = np.linalg.norm(hand_pos - waypoint)
            reward -= dist_to_waypoint 
        
        # 4. 最终任务奖励：球靠近目标
        if dist_h2b < 0.05: # 如果手抓住了球 (简化判断)
            reward += 1.0 # 抓取奖励
            reward -= dist_b2t * 2.0 # 运送奖励权重翻倍
            
        return reward

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        return self.get_obs(), self.compute_reward(), False

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # 可以在这里随机球的位置
        return self.get_obs()