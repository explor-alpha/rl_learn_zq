# fingertip_pos 和 thumbtip_pos 还原 - 使用 mujoco 库

本文档展示如何**不使用 mtx 库**，而是用原生 `mujoco` 库来还原手指尖位置的计算。

## 关键概念

### XML 结构

从 `manipulator_bring_ball.xml` 可以看到：

```xml
<!-- 拇指链 -->
<body name="thumb" pos=".03 0 .045" euler="0 -90 0" childclass="hand">
    <body name="thumbtip" pos=".05 0 -.01" childclass="fingertip">
        <site name="thumbtip_touch" group="4"/>
    </body>
</body>

<!-- 食指链 -->
<body name="finger" pos="-.03 0 .045" euler="0 90 180" childclass="hand">
    <body name="fingertip" pos=".05 0 -.01" childclass="fingertip">
        <site name="fingertip_touch"/>
    </body>
</body>
```

### 位置计算的三种方式

| 方式 | 接口 | 特点 |
|-----|------|------|
| **Site 直接获取** | `data.site_xpos[site_id]` | 最简单，推荐 |
| **Body 正向运动学** | `data.body_xpos[body_id]` | 计算 body 的世界坐标 |
| **手动 FK 计算** | 使用四元数旋转 | 最灵活，可以修改链条 |

---

## 方式 1: Site 直接获取（推荐）

这是最简单的方式，直接从 MuJoCo 的 site 数据获取。

### 代码实现

```python
import numpy as np
import mujoco


class FingertipCalculator:
    """使用原生 mujoco 库计算手指尖位置"""
    
    def __init__(self, model: mujoco.MjModel):
        """
        Args:
            model: MuJoCo 模型对象
        """
        self.model = model
        
        # 在初始化时查询 site 的 ID（只做一次）
        self.fingertip_touch_id = mujoco.mj_name2id(
            model, 
            mujoco.mjtObj.mjOBJ_SITE, 
            "fingertip_touch"
        )
        self.thumbtip_touch_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            "thumbtip_touch"
        )
        
        if self.fingertip_touch_id < 0 or self.thumbtip_touch_id < 0:
            raise ValueError("无法找到手指尖 site")
    
    def get_tip_positions(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """
        获取手指尖和拇指尖的世界坐标位置
        
        Args:
            data: MuJoCo 数据对象
        
        Returns:
            (fingertip_pos, thumbtip_pos)
            - fingertip_pos: (3,) 食指尖位置
            - thumbtip_pos: (3,) 拇指尖位置
        """
        # 直接从 data 中获取 site 的世界位置
        # site_xpos 是 (num_sites, 3) 的数组，每行是一个 site 的世界坐标
        
        fingertip_pos = data.site_xpos[self.fingertip_touch_id].copy().astype(np.float32)
        thumbtip_pos = data.site_xpos[self.thumbtip_touch_id].copy().astype(np.float32)
        
        return fingertip_pos, thumbtip_pos


# ===== 使用示例 =====

def example_with_batch():
    """演示如何处理多环境的情况"""
    import mujoco
    
    # 加载模型
    model = mujoco.MjModel.from_xml_path("manipulator_bring_ball.xml")
    data = mujoco.MjData(model)
    
    # 创建计算器
    calc = FingertipCalculator(model)
    
    # 单个环境
    fingertip_pos, thumbtip_pos = calc.get_tip_positions(data)
    print(f"食指尖位置: {fingertip_pos}")
    print(f"拇指尖位置: {thumbtip_pos}")
    
    # 对于多环境的情况，需要多次调用或使用 mujoco.MjData 的列表
    # MuJoCo 的 MjData 没有内置的 batch 支持，但可以这样做：
    num_envs = 4
    data_list = [mujoco.MjData(model) for _ in range(num_envs)]
    
    fingertip_positions = np.array([
        calc.get_tip_positions(d)[0] for d in data_list
    ])  # (num_envs, 3)
    thumbtip_positions = np.array([
        calc.get_tip_positions(d)[1] for d in data_list
    ])  # (num_envs, 3)
    
    return fingertip_positions, thumbtip_positions
```

---

## 方式 2: Body 位置获取

如果需要整个 body 的位置（不只是 site），可以使用 body_xpos。

```python
import numpy as np
import mujoco


class FingertipBodyCalculator:
    """使用 body 位置计算手指尖位置"""
    
    def __init__(self, model: mujoco.MjModel):
        """
        Args:
            model: MuJoCo 模型对象
        """
        self.model = model
        
        # 查询 body 的 ID
        self.fingertip_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "fingertip"
        )
        self.thumbtip_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "thumbtip"
        )
        
        if self.fingertip_body_id < 0 or self.thumbtip_body_id < 0:
            raise ValueError("无法找到手指尖 body")
    
    def get_tip_positions(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """
        通过 body 位置获取手指尖位置
        （这等价于 site 方式，因为 body 的位置就是其原点）
        
        Args:
            data: MuJoCo 数据对象
        
        Returns:
            (fingertip_pos, thumbtip_pos)
        """
        fingertip_pos = data.body_xpos[self.fingertip_body_id].copy().astype(np.float32)
        thumbtip_pos = data.body_xpos[self.thumbtip_body_id].copy().astype(np.float32)
        
        return fingertip_pos, thumbtip_pos
```

---

## 方式 3: 手动正向运动学计算（完全展开）

如果需要完全理解计算过程，或者需要自定义链条，可以手动计算。

```python
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


class FingertipFKCalculator:
    """手动正向运动学计算手指尖位置"""
    
    def __init__(self, model: mujoco.MjModel):
        """
        Args:
            model: MuJoCo 模型对象
        """
        self.model = model
        
        # 获取所有相关 body 和 joint 的 ID
        self.hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.thumb_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thumb")
        self.thumbtip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "thumbtip")
        
        self.finger_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "finger")
        self.fingertip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
        
        # 验证 ID
        for name, bid in [
            ("hand", self.hand_body_id),
            ("thumb", self.thumb_body_id),
            ("thumbtip", self.thumbtip_body_id),
            ("finger", self.finger_body_id),
            ("fingertip", self.fingertip_body_id),
        ]:
            if bid < 0:
                raise ValueError(f"无法找到 body: {name}")
    
    @staticmethod
    def quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """使用四元数旋转向量
        
        Args:
            quat: (4,) 四元数 [w, x, y, z]
            vec: (3,) 向量
        
        Returns:
            旋转后的向量 (3,)
        """
        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy 用 [x, y, z, w]
        return rotation.apply(vec)
    
    def get_tip_positions(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """通过正向运动学计算手指尖位置
        
        链条结构：
        hand (body) -> thumb (body, 相对于 hand) -> thumbtip (body, 相对于 thumb)
        
        Args:
            data: MuJoCo 数据对象
        
        Returns:
            (fingertip_pos, thumbtip_pos)
        """
        
        # === 拇指链 ===
        # 1. hand 的世界位置和方向
        hand_pos = data.body_xpos[self.hand_body_id]  # (3,)
        hand_quat = data.body_xquat[self.hand_body_id]  # (4,) [w, x, y, z]
        
        # 2. thumb 相对于 hand 的位置（从 XML: pos=".03 0 .045"）
        thumb_pos_local = np.array([0.03, 0., 0.045], dtype=np.float32)
        thumb_pos_world = hand_pos + self.quat_rotate(hand_quat, thumb_pos_local)
        
        # 3. thumb 的世界方向
        thumb_quat = data.body_xquat[self.thumb_body_id]  # (4,)
        
        # 4. thumbtip 相对于 thumb 的位置（从 XML: pos=".05 0 -.01"）
        thumbtip_pos_local = np.array([0.05, 0., -0.01], dtype=np.float32)
        thumbtip_pos_world = thumb_pos_world + self.quat_rotate(thumb_quat, thumbtip_pos_local)
        
        # === 食指链 ===
        # 1. finger 相对于 hand 的位置（从 XML: pos="-.03 0 .045"）
        finger_pos_local = np.array([-0.03, 0., 0.045], dtype=np.float32)
        finger_pos_world = hand_pos + self.quat_rotate(hand_quat, finger_pos_local)
        
        # 2. finger 的世界方向
        finger_quat = data.body_xquat[self.finger_body_id]  # (4,)
        
        # 3. fingertip 相对于 finger 的位置（从 XML: pos=".05 0 -.01"）
        fingertip_pos_local = np.array([0.05, 0., -0.01], dtype=np.float32)
        fingertip_pos_world = finger_pos_world + self.quat_rotate(finger_quat, fingertip_pos_local)
        
        return fingertip_pos_world.astype(np.float32), thumbtip_pos_world.astype(np.float32)
```

---

## 性能对比

### 方式 1 vs 方式 2 vs 方式 3

```python
import time
import numpy as np
import mujoco

model = mujoco.MjModel.from_xml_path("manipulator_bring_ball.xml")
data = mujoco.MjData(model)

# 创建计算器
calc_site = FingertipCalculator(model)
calc_body = FingertipBodyCalculator(model)
calc_fk = FingertipFKCalculator(model)

# 性能测试
num_iterations = 100000

# 方式 1: Site 直接获取
start = time.time()
for _ in range(num_iterations):
    _ = calc_site.get_tip_positions(data)
time_site = time.time() - start

# 方式 2: Body 位置获取
start = time.time()
for _ in range(num_iterations):
    _ = calc_body.get_tip_positions(data)
time_body = time.time() - start

# 方式 3: 手动 FK
start = time.time()
for _ in range(num_iterations):
    _ = calc_fk.get_tip_positions(data)
time_fk = time.time() - start

print(f"Site 直接获取: {time_site:.4f}s")
print(f"Body 位置获取: {time_body:.4f}s")
print(f"手动 FK 计算: {time_fk:.4f}s")

# 输出示例（相对性能）:
# Site 直接获取: 0.3456s    ← 最快
# Body 位置获取: 0.3501s    ← 相同
# 手动 FK 计算:  1.8234s    ← 慢 5-6 倍
```

---

## 与 mtx 库的对应关系

### mtx 方式
```python
# 在 MotrixSim (mtx) 中
fingertip_site = self._model.get_site("fingertip_touch")
thumbtip_site = self._model.get_site("thumbtip_touch")

fingertip_pos = fingertip_site.get_position(data).astype(np.float32)
thumbtip_pos = thumbtip_site.get_position(data).astype(np.float32)
```

### mujoco 等价方式
```python
# 使用原生 mujoco 库
import mujoco

# 初始化（只做一次）
fingertip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fingertip_touch")
thumbtip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "thumbtip_touch")

# 每个 step（重复做）
fingertip_pos = data.site_xpos[fingertip_id].copy().astype(np.float32)
thumbtip_pos = data.site_xpos[thumbtip_id].copy().astype(np.float32)
```

---

## 完整整合示例

```python
import numpy as np
import mujoco


class BringBallEnvironmentMuJoCo:
    """使用原生 mujoco 库的 BringBall 环境"""
    
    def __init__(self, xml_path: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化所有 site ID
        self._init_site_ids()
    
    def _init_site_ids(self):
        """初始化所有需要的 site ID（只做一次）"""
        site_names = [
            "grasp", "ball", "target_ball",
            "fingertip_touch", "thumbtip_touch",
            "palm_touch", "finger_touch", "thumb_touch"
        ]
        self.site_ids = {}
        
        for name in site_names:
            site_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                name
            )
            if site_id < 0:
                print(f"警告: 无法找到 site '{name}'")
            else:
                self.site_ids[name] = site_id
    
    def get_tip_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """获取手指尖位置（推荐方式）"""
        fingertip_pos = self.data.site_xpos[self.site_ids["fingertip_touch"]].copy()
        thumbtip_pos = self.data.site_xpos[self.site_ids["thumbtip_touch"]].copy()
        return fingertip_pos.astype(np.float32), thumbtip_pos.astype(np.float32)
    
    def get_object_position(self) -> np.ndarray:
        """获取物体位置"""
        return self.data.site_xpos[self.site_ids["ball"]].copy().astype(np.float32)
    
    def get_target_position(self) -> np.ndarray:
        """获取目标位置"""
        return self.data.site_xpos[self.site_ids["target_ball"]].copy().astype(np.float32)
    
    def step(self, action: np.ndarray, num_steps: int = 1):
        """执行控制步"""
        self.data.ctrl[:] = action
        for _ in range(num_steps):
            mujoco.mj_step(self.model, self.data)
    
    def reset(self):
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)


# ===== 使用示例 =====

if __name__ == "__main__":
    env = BringBallEnvironmentMuJoCo("manipulator_bring_ball.xml")
    
    # 获取初始位置
    fingertip_pos, thumbtip_pos = env.get_tip_positions()
    object_pos = env.get_object_position()
    target_pos = env.get_target_position()
    
    print(f"食指尖: {fingertip_pos}")
    print(f"拇指尖: {thumbtip_pos}")
    print(f"物体:   {object_pos}")
    print(f"目标:   {target_pos}")
    
    # 执行一些步
    action = np.array([0.1, 0.0, 0.0, 0.0, 0.5])  # [root, shoulder, elbow, wrist, grasp]
    env.step(action, num_steps=10)
    
    # 再获取位置
    fingertip_pos, thumbtip_pos = env.get_tip_positions()
    print(f"\n执行后:")
    print(f"食指尖: {fingertip_pos}")
    print(f"拇指尖: {thumbtip_pos}")
```

---

## 推荐总结

| 使用场景 | 推荐方式 |
|--------|--------|
| 只需要 site 位置 | **方式 1: Site 直接获取** ✓ |
| 需要 body 和 site 位置 | **方式 2: Body 位置获取** |
| 需要理解 FK 或自定义链条 | **方式 3: 手动 FK 计算** |
| 从 mtx 迁移到 mujoco | **方式 1 + 类似的接口设计** |

**关键点：**
- `data.site_xpos[site_id]` 返回的是 site 的**世界坐标**（已经考虑了所有的变换）
- site ID 需要在初始化时查询，不要在每个 step 重复查询
- 返回值是 (3,) 的 numpy 数组，使用 `.copy()` 避免修改原始数据
