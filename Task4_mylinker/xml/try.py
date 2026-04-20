import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class FragileGraspEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 加载刚才创建的 scene.xml
        self.model = mujoco.MjModel.from_xml_path('scene.xml')
        self.data = mujoco.MjData(self.model)
        
        # 6个主控关节 ID
        self.active_joint_ids = [1, 0, 3, 5, 7, 9] 
        self.f_max = 5.0  # 定义破碎阈值：超过 5N 就碎了
        
        # Action: [-1, 1] 的残差位置控制
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        # Observation: 本体感知 + 物体位置 (简化版)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

    def step(self, action):
        # 1. 动作映射：将 [-1, 1] 映射到小的位移量
        step_size = 0.05
        target_qpos = self.data.qpos[self.active_joint_ids] + action * step_size
        self.data.ctrl[self.active_joint_ids] = target_qpos
        
        # 2. 物理仿真步
        mujoco.mj_step(self.model, self.data)
        
        # 3. 接触力检测（特权信息）
        total_force = self._get_contact_force()
        
        # 4. 判断状态
        broken = total_force > self.f_max
        success = self.data.body('fragile_object').xpos[2] > 0.1 # 提到 10cm 高度算成功
        
        # 5. 计算奖励
        reward = self._compute_reward(total_force, broken, success)
        
        # 如果碎了，直接结束
        terminated = broken or success
        obs = self._get_obs()
        
        return obs, reward, terminated, False, {"force": total_force}

    def _get_contact_force(self):
        """获取物体受到的所有合力"""
        force = 0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # 检查碰撞体中是否包含物体
            if contact.geom1 == mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom") or \
               contact.geom2 == mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"):
                # 获取法向接触力
                c_res = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, c_res)
                force += np.abs(c_res[0]) # 这里的 0 是法向力
        return force

    def _compute_reward(self, force, broken, success):
        if broken:
            return -100.0  # 碎了给大惩罚
        if success:
            return 200.0   # 成功抓起给大奖励
        
        # 距离奖励：手掌靠近物体
        hand_pos = self.data.body('hand_base_link').xpos
        obj_pos = self.data.body('fragile_object').xpos
        dist = np.linalg.norm(hand_pos - obj_pos)
        
        reward = -dist # 越近奖励越高
        
        # 温柔奖励：如果有力，但没碎，给一个小惩罚鼓励更轻的力
        if force > 0.1:
            reward -= 0.1 * force
            
        return reward

    def _get_obs(self):
        # 拼接关节位置、速度和物体相对位置
        return np.concatenate([
            self.data.qpos[self.active_joint_ids],
            self.data.body('fragile_object').xpos - self.data.body('hand_base_link').xpos
        ])