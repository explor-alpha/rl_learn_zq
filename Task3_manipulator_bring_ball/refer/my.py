import mujoco
import numpy as np

class ManipulatorEnv:
    def __init__(self, model_path="model.xml"):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
    def get_obs(self):
        # 获取所有关节位置和速度
        return np.concatenate([self.data.qpos, self.data.qvel])

    def compute_reward(self):
        """
        这里是【魔改奖励函数】的核心区域
        """
        # 1. 获取各个物体的坐标
        hand_pos = self.data.site("pinch").xpos  # 夹爪位置
        ball_pos = self.data.body("ball").xpos   # 球的位置
        target_pos = self.data.mocap_pos[0]      # 目标位置
        wall_x = 0.2                             # 格挡的 X 坐标
        wall_top_z = 0.4                         # 格挡的顶部高度

        # 2. 计算基础距离
        dist_hand_to_ball = np.linalg.norm(hand_pos - ball_pos)
        dist_ball_to_target = np.linalg.norm(ball_pos - target_pos)

        # 3. 绕路逻辑 (Waypoints)
        # 如果手和球被墙隔开了 (手在左，球在右)
        reward = 0
        if hand_pos[0] < wall_x and ball_pos[0] > wall_x:
            # 引导点设为墙的上方边缘
            waypoint = np.array([wall_x, 0, wall_top_z + 0.1])
            dist_to_waypoint = np.linalg.norm(hand_pos - waypoint)
            reward = -dist_to_waypoint # 奖励是靠近引导点
        else:
            # 已经过墙了，奖励是靠近球
            reward = -dist_hand_to_ball
        
        # 4. 抓取奖励
        # 如果球被举起来了，或者球靠近目标，给大奖
        if ball_pos[2] > 0.05: # 球离开地面
            reward += 1.0
        
        reward -= 0.5 * dist_ball_to_target # 最终目标奖励
        
        return reward

    def step(self, action):
        # 输入动作 (5维: root, shoulder, elbow, wrist, grasp)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        
        obs = self.get_obs()
        reward = self.compute_reward()
        done = False # 可根据步数或球掉落自定义
        return obs, reward, done

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self.get_obs()