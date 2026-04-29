"""
使用原生 mujoco 库还原 fingertip_pos 和 thumbtip_pos
不依赖 mtx 库的实现
"""

import numpy as np
import mujoco
from typing import Tuple


# ============================================================================
# 方式 1: Site 直接获取（推荐，性能最好）
# ============================================================================

class FingertipPositionCalculator:
    """使用 mujoco 库直接从 site 获取手指尖位置"""
    
    def __init__(self, model: mujoco.MjModel):
        """
        初始化计算器，查询所有必要的 site ID
        
        Args:
            model: MuJoCo 模型对象
        """
        self.model = model
        self._init_site_ids()
    
    def _init_site_ids(self):
        """查询并缓存所有必要的 site ID（只在初始化时做一次）"""
        site_names = {
            'fingertip_touch': "fingertip_touch",
            'thumbtip_touch': "thumbtip_touch",
            'grasp': "grasp",
            'ball': "ball",
            'target_ball': "target_ball",
            'palm_touch': "palm_touch",
            'finger_touch': "finger_touch",
            'thumb_touch': "thumb_touch",
        }
        
        self.site_ids = {}
        for key, name in site_names.items():
            site_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                name
            )
            if site_id >= 0:
                self.site_ids[key] = site_id
            else:
                print(f"⚠ 警告: 无法找到 site '{name}'")
    
    def get_tip_positions(self, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取手指尖和拇指尖的世界坐标
        
        Args:
            data: MuJoCo 数据对象（包含当前的物理状态）
        
        Returns:
            (fingertip_pos, thumbtip_pos)
            - fingertip_pos: (3,) float32，食指尖的世界坐标
            - thumbtip_pos: (3,) float32，拇指尖的世界坐标
        """
        fingertip_pos = data.site_xpos[self.site_ids['fingertip_touch']].copy()
        thumbtip_pos = data.site_xpos[self.site_ids['thumbtip_touch']].copy()
        
        return fingertip_pos.astype(np.float32), thumbtip_pos.astype(np.float32)
    
    def get_grasp_position(self, data: mujoco.MjData) -> np.ndarray:
        """获取手部抓握点位置"""
        grasp_pos = data.site_xpos[self.site_ids['grasp']].copy()
        return grasp_pos.astype(np.float32)
    
    def get_object_position(self, data: mujoco.MjData) -> np.ndarray:
        """获取物体位置"""
        ball_pos = data.site_xpos[self.site_ids['ball']].copy()
        return ball_pos.astype(np.float32)
    
    def get_target_position(self, data: mujoco.MjData) -> np.ndarray:
        """获取目标位置"""
        target_pos = data.site_xpos[self.site_ids['target_ball']].copy()
        return target_pos.astype(np.float32)


# ============================================================================
# 方式 2: Body 位置获取
# ============================================================================

class FingertipBodyCalculator:
    """使用 body 位置计算手指尖位置（等价于方式1，但展示另一种方法）"""
    
    def __init__(self, model: mujoco.MjModel):
        """初始化，查询 body ID"""
        self.model = model
        
        # 查询 body ID
        self.fingertip_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "fingertip"
        )
        self.thumbtip_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "thumbtip"
        )
        
        if self.fingertip_body_id < 0 or self.thumbtip_body_id < 0:
            raise ValueError("无法找到 fingertip 或 thumbtip body")
    
    def get_tip_positions(self, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
        """
        通过 body 位置获取手指尖位置
        
        说明：
        body 的 xpos 是该 body 的原点（frame origin）在世界坐标系中的位置。
        对于 fingertip/thumbtip body，原点就在手指尖处。
        """
        fingertip_pos = data.body_xpos[self.fingertip_body_id].copy()
        thumbtip_pos = data.body_xpos[self.thumbtip_body_id].copy()
        
        return fingertip_pos.astype(np.float32), thumbtip_pos.astype(np.float32)


# ============================================================================
# 方式 3: 手动正向运动学计算
# ============================================================================

class FingertipFKCalculator:
    """
    手动正向运动学计算手指尖位置
    展示完整的 FK 链条，可以用于理解或自定义
    """
    
    def __init__(self, model: mujoco.MjModel):
        """初始化，查询所有相关 body ID"""
        self.model = model
        
        # 获取所有 body ID
        self.hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.thumb_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thumb")
        self.thumbtip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thumbtip")
        self.finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger")
        self.fingertip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
        
        # 验证
        body_ids = {
            "hand": self.hand_body_id,
            "thumb": self.thumb_body_id,
            "thumbtip": self.thumbtip_body_id,
            "finger": self.finger_body_id,
            "fingertip": self.fingertip_body_id,
        }
        
        for name, bid in body_ids.items():
            if bid < 0:
                raise ValueError(f"无法找到 body: {name}")
        
        # 从 XML 中的相对位置定义
        # <body name="thumb" pos=".03 0 .045" ...>
        # <body name="thumbtip" pos=".05 0 -.01" ...>
        self.thumb_rel_pos = np.array([0.03, 0., 0.045], dtype=np.float32)
        self.thumbtip_rel_pos = np.array([0.05, 0., -0.01], dtype=np.float32)
        
        # <body name="finger" pos="-.03 0 .045" ...>
        # <body name="fingertip" pos=".05 0 -.01" ...>
        self.finger_rel_pos = np.array([-0.03, 0., 0.045], dtype=np.float32)
        self.fingertip_rel_pos = np.array([0.05, 0., -0.01], dtype=np.float32)
    
    @staticmethod
    def quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """
        使用四元数旋转向量
        
        Args:
            quat: (4,) 四元数 [w, x, y, z]（MuJoCo 格式）
            vec: (3,) 向量
        
        Returns:
            旋转后的向量 (3,)
        """
        from scipy.spatial.transform import Rotation
        
        # scipy 使用 [x, y, z, w]，MuJoCo 使用 [w, x, y, z]
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        rotation = Rotation.from_quat(quat_scipy)
        return rotation.apply(vec)
    
    def get_tip_positions(self, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
        """
        通过正向运动学计算手指尖位置
        
        FK 链条：
        hand -> thumb -> thumbtip
        hand -> finger -> fingertip
        """
        
        # ===== 拇指链 =====
        # 1. hand 的世界位置和姿态
        hand_pos = data.body_xpos[self.hand_body_id]  # (3,)
        hand_quat = data.body_xquat[self.hand_body_id]  # (4,) [w, x, y, z]
        
        # 2. thumb 的世界位置 = hand_pos + R(hand_quat) * thumb_rel_pos
        thumb_pos = hand_pos + self.quat_rotate(hand_quat, self.thumb_rel_pos)
        
        # 3. thumb 的世界姿态
        thumb_quat = data.body_xquat[self.thumb_body_id]  # (4,)
        
        # 4. thumbtip 的世界位置 = thumb_pos + R(thumb_quat) * thumbtip_rel_pos
        thumbtip_pos = thumb_pos + self.quat_rotate(thumb_quat, self.thumbtip_rel_pos)
        
        # ===== 食指链 =====
        # 1. finger 的世界位置 = hand_pos + R(hand_quat) * finger_rel_pos
        finger_pos = hand_pos + self.quat_rotate(hand_quat, self.finger_rel_pos)
        
        # 2. finger 的世界姿态
        finger_quat = data.body_xquat[self.finger_body_id]  # (4,)
        
        # 3. fingertip 的世界位置 = finger_pos + R(finger_quat) * fingertip_rel_pos
        fingertip_pos = finger_pos + self.quat_rotate(finger_quat, self.fingertip_rel_pos)
        
        return fingertip_pos.astype(np.float32), thumbtip_pos.astype(np.float32)


# ============================================================================
# 完整环境类
# ============================================================================

class BringBallEnvironmentMuJoCo:
    """
    使用原生 mujoco 库的 BringBall 环境
    完全不依赖 mtx 库
    """
    
    def __init__(self, xml_path: str):
        """
        初始化环境
        
        Args:
            xml_path: MJCF XML 文件路径
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 使用方式 1 的计算器（推荐）
        self.calculator = FingertipPositionCalculator(self.model)
        
        # 获取一些基本信息
        self.num_actuators = self.model.nu
        self.num_dofs = self.model.nv
    
    def get_state(self) -> dict:
        """获取环境的完整状态"""
        fingertip_pos, thumbtip_pos = self.calculator.get_tip_positions(self.data)
        grasp_pos = self.calculator.get_grasp_position(self.data)
        object_pos = self.calculator.get_object_position(self.data)
        target_pos = self.calculator.get_target_position(self.data)
        
        # 计算距离
        dist_finger = np.linalg.norm(fingertip_pos - object_pos)
        dist_thumb = np.linalg.norm(thumbtip_pos - object_pos)
        avg_tip_dist = (dist_finger + dist_thumb) / 2.0
        move_dist = np.linalg.norm(object_pos - target_pos)
        
        return {
            'fingertip_pos': fingertip_pos,
            'thumbtip_pos': thumbtip_pos,
            'grasp_pos': grasp_pos,
            'object_pos': object_pos,
            'target_pos': target_pos,
            'avg_tip_dist': avg_tip_dist,
            'move_dist': move_dist,
            'dof_pos': self.data.qpos.copy(),
            'dof_vel': self.data.qvel.copy(),
        }
    
    def step(self, action: np.ndarray, num_steps: int = 1) -> None:
        """
        执行控制步
        
        Args:
            action: (num_actuators,) 控制命令
            num_steps: 执行多少个物理步
        """
        self.data.ctrl[:] = action
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)
    
    def reset(self) -> None:
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)
    
    def render(self, width: int = 480, height: int = 480) -> np.ndarray:
        """
        渲染环境视图
        
        Args:
            width: 图像宽度
            height: 图像高度
        
        Returns:
            RGB 图像数组 (height, width, 3)
        """
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data)
        image = renderer.render()
        return image


# ============================================================================
# 测试和演示
# ============================================================================

if __name__ == "__main__":
    import os
    
    # 构造 XML 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(
        script_dir,
        "motrix_envs/src/motrix_envs/basic/manipulator/manipulator_bring_ball.xml"
    )
    
    if not os.path.exists(xml_path):
        print(f"✗ 文件不存在: {xml_path}")
        print("\n请调整 xml_path 指向正确的 XML 文件路径")
        exit(1)
    
    print("=" * 70)
    print("BringBall 环境 - MuJoCo 原生库还原")
    print("=" * 70)
    
    # 初始化环境
    env = BringBallEnvironmentMuJoCo(xml_path)
    
    print("\n✓ 环境初始化成功")
    print(f"  - 执行器数量: {env.num_actuators}")
    print(f"  - DOF 数量: {env.num_dofs}")
    
    # 获取初始状态
    state = env.get_state()
    print("\n【初始状态】")
    print(f"食指尖位置: {state['fingertip_pos']}")
    print(f"拇指尖位置: {state['thumbtip_pos']}")
    print(f"手部位置: {state['grasp_pos']}")
    print(f"物体位置: {state['object_pos']}")
    print(f"目标位置: {state['target_pos']}")
    print(f"平均手指距离: {state['avg_tip_dist']:.4f}")
    print(f"物体到目标距离: {state['move_dist']:.4f}")
    
    # 执行几个控制步
    print("\n【执行控制】")
    action = np.array([0.1, 0.05, -0.1, 0.0, 0.3])  # [root, shoulder, elbow, wrist, grasp]
    print(f"控制命令: {action}")
    env.step(action, num_steps=10)
    
    # 再获取状态
    state = env.get_state()
    print("\n【执行后状态】")
    print(f"食指尖位置: {state['fingertip_pos']}")
    print(f"拇指尖位置: {state['thumbtip_pos']}")
    print(f"物体位置: {state['object_pos']}")
    print(f"平均手指距离: {state['avg_tip_dist']:.4f}")
    print(f"物体到目标距离: {state['move_dist']:.4f}")
    
    print("\n✓ 测试完成")
    
    # 性能测试
    print("\n【性能测试 - 100000 次迭代】")
    import time
    
    calc1 = FingertipPositionCalculator(env.model)
    calc2 = FingertipBodyCalculator(env.model)
    calc3 = FingertipFKCalculator(env.model)
    
    # 方式 1
    start = time.time()
    for _ in range(100000):
        _ = calc1.get_tip_positions(env.data)
    time1 = time.time() - start
    print(f"方式1 (Site 直接获取): {time1:.4f}s")
    
    # 方式 2
    start = time.time()
    for _ in range(100000):
        _ = calc2.get_tip_positions(env.data)
    time2 = time.time() - start
    print(f"方式2 (Body 位置获取): {time2:.4f}s")
    
    # 方式 3
    start = time.time()
    for _ in range(100000):
        _ = calc3.get_tip_positions(env.data)
    time3 = time.time() - start
    print(f"方式3 (手动 FK 计算):  {time3:.4f}s")
    
    print(f"\n相对速度: {time3/time1:.1f}x 倍（方式3 vs 方式1）")
