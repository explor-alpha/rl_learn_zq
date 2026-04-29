# mtx 库 vs mujoco 库 - fingertip_pos 还原对照

快速参考：如何将 mtx 库的代码迁移到原生 mujoco 库

## 一览表

| 操作 | mtx 库 | mujoco 库 |
|-----|--------|-----------|
| 加载模型 | `mtx.load_model(path)` | `mujoco.MjModel.from_xml_path(path)` |
| 创建数据 | (自动) | `mujoco.MjData(model)` |
| 获取 Site ID | (内部处理) | `mujoco.mj_name2id(model, mjOBJ_SITE, name)` |
| 获取 Site 位置 | `site.get_position(data)` | `data.site_xpos[site_id]` |
| 获取 Body ID | (内部处理) | `mujoco.mj_name2id(model, mjOBJ_BODY, name)` |
| 获取 Body 位置 | (不直接支持) | `data.body_xpos[body_id]` |
| 执行一步 | `model.step(data)` | `mujoco.mj_step(model, data)` |
| 重置 | (方法不同) | `mujoco.mj_resetData(model, data)` |
| 获取关节位置 | `data.dof_pos` | `data.qpos` |
| 获取关节速度 | `data.dof_vel` | `data.qvel` |
| 获取接触 | `model.get_contact_query(data)` | `data.contact` |

---

## 代码对比

### 1. 初始化

#### mtx 方式
```python
import motrixsim as mtx

model = mtx.load_model("manipulator_bring_ball.xml")
data = mtx.SceneData.create(model)

grasp_site = model.get_site("grasp")
fingertip_site = model.get_site("fingertip_touch")
thumbtip_site = model.get_site("thumbtip_touch")
```

#### mujoco 方式
```python
import mujoco

model = mujoco.MjModel.from_xml_path("manipulator_bring_ball.xml")
data = mujoco.MjData(model)

# Site 初始化
fingertip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fingertip_touch")
thumbtip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "thumbtip_touch")
grasp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp")
```

---

### 2. 获取位置

#### mtx 方式
```python
# 直接调用方法
fingertip_pos = fingertip_site.get_position(data)
thumbtip_pos = thumbtip_site.get_position(data)
grasp_pos = grasp_site.get_position(data)
```

#### mujoco 方式
```python
# 通过数据数组索引
fingertip_pos = data.site_xpos[fingertip_id]
thumbtip_pos = data.site_xpos[thumbtip_id]
grasp_pos = data.site_xpos[grasp_id]

# 如果需要 float32 和独立副本
fingertip_pos = data.site_xpos[fingertip_id].copy().astype(np.float32)
```

---

### 3. 执行控制步

#### mtx 方式
```python
data.actuator_ctrls = action
model.step(data)
```

#### mujoco 方式
```python
data.ctrl[:] = action
mujoco.mj_step(model, data)

# 多步执行
for _ in range(num_steps):
    mujoco.mj_step(model, data)
```

---

### 4. 获取关节数据

#### mtx 方式
```python
# DOF 位置和速度
qpos = data.dof_pos
qvel = data.dof_vel

# 特定关节
arm_pos = data.dof_pos[:, arm_dof_pos_indices]
arm_vel = data.dof_vel[:, arm_dof_vel_indices]
```

#### mujoco 方式
```python
# 关节位置和速度（单环境）
qpos = data.qpos  # (num_qpos,)
qvel = data.qvel  # (num_qvel,)

# 特定关节
arm_pos = data.qpos[arm_qpos_indices]
arm_vel = data.qvel[arm_qvel_indices]

# 注意：mujoco.MjData 默认是单个环境
# 对于多环境，需要创建多个 MjData 或使用向量化库
```

---

### 5. 获取传感器值

#### mtx 方式
```python
touch_value = model.get_sensor_value("palm_touch", data)
```

#### mujoco 方式
```python
# 方法 1: 如果 sensor 有输出缓冲
sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "palm_touch")
touch_value = data.sensordata[model.sensor_adr[sensor_id]]

# 方法 2: 通过接触查询
contact_query = mujoco.MjData(model)
# 遍历 data.contact 获取接触信息
```

---

### 6. 完整的环境更新流程

#### mtx 方式
```python
class BringBallMtx:
    def __init__(self, cfg):
        self._model = mtx.load_model(cfg.model_file)
        self._grasp_site = self._model.get_site("grasp")
        self._fingertip_site = self._model.get_site("fingertip_touch")
        # ... 更多初始化
    
    def update_state(self, data):
        # 获取位置
        object_pos = self._object_site.get_position(data)
        fingertip_pos = self._fingertip_site.get_position(data)
        thumbtip_pos = self._thumbtip_site.get_position(data)
        
        # 计算距离
        dist_finger = np.linalg.norm(fingertip_pos - object_pos)
        dist_thumb = np.linalg.norm(thumbtip_pos - object_pos)
        avg_tip_dist = (dist_finger + dist_thumb) / 2.0
        
        return avg_tip_dist
```

#### mujoco 方式
```python
class BringBallMuJoCo:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 初始化 site ID
        self.grasp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp")
        self.fingertip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fingertip_touch")
        self.thumbtip_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "thumbtip_touch")
        self.object_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ball")
    
    def update_state(self):
        # 获取位置
        object_pos = self.data.site_xpos[self.object_id]
        fingertip_pos = self.data.site_xpos[self.fingertip_id]
        thumbtip_pos = self.data.site_xpos[self.thumbtip_id]
        
        # 计算距离
        dist_finger = np.linalg.norm(fingertip_pos - object_pos)
        dist_thumb = np.linalg.norm(thumbtip_pos - object_pos)
        avg_tip_dist = (dist_finger + dist_thumb) / 2.0
        
        return avg_tip_dist
```

---

## 关键差异

### 1. 多环境支持

**mtx:**
```python
# 原生支持多环境（向量化）
data.shape[0]  # 环境数量
# data 包含所有环境的数据
```

**mujoco:**
```python
# 默认单环境
# 多环境需要显式管理
data_list = [mujoco.MjData(model) for _ in range(num_envs)]
# 或使用专门的向量化库
```

### 2. 数据访问

**mtx:**
```python
# 返回对象，可以调用方法
pos = site.get_position(data)  # 返回 numpy 数组，已经转换
```

**mujoco:**
```python
# 直接访问数组，返回视图或数据指针
pos = data.site_xpos[site_id]  # 返回视图，可能需要 .copy()
```

### 3. 关节索引

**mtx:**
```python
# 通过名称直接映射
arm_dof_pos_indices = model.joint_dof_pos_indices[arm_joint_indices]
```

**mujoco:**
```python
# 需要查询关节的 qpos_adr 和 qvel_adr
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "arm_shoulder")
qpos_idx = model.jnt_qposadr[joint_id]
qvel_idx = model.jnt_dofadr[joint_id]
```

---

## 迁移检查清单

- [ ] 替换 `mtx.load_model()` → `mujoco.MjModel.from_xml_path()`
- [ ] 替换 `mtx.SceneData` → `mujoco.MjData()`
- [ ] 缓存所有 name2id 查询结果（在 __init__ 中）
- [ ] 替换 `.get_position(data)` → `data.site_xpos[id]`
- [ ] 替换 `.get_pose(data)` → `data.site_xpos[id]` + `data.site_xmat[id]` (或 quaternion)
- [ ] 处理多环境的数据结构差异
- [ ] 使用 `.copy()` 避免修改底层数组
- [ ] 使用 `mujoco.mj_step()` 而不是 `model.step()`

---

## 性能提示

1. **缓存 ID 查询**：在初始化时做一次，不要在每个 step 重复
   ```python
   # ✓ 好
   site_id = mujoco.mj_name2id(...)  # init
   pos = data.site_xpos[site_id]      # step
   
   # ✗ 差
   site_id = mujoco.mj_name2id(...)  # step (每次都查询)
   pos = data.site_xpos[site_id]
   ```

2. **避免不必要的 copy**：只在需要修改或避免竞态条件时使用
   ```python
   # ✓ 好（只读）
   pos = data.site_xpos[site_id]
   dist = np.linalg.norm(pos - target)
   
   # ✓ 好（需要修改）
   pos = data.site_xpos[site_id].copy()
   pos += offset
   ```

3. **批量操作**：使用 numpy 向量化而不是循环
   ```python
   # ✓ 好
   for site_id in site_ids:
       positions[i] = data.site_xpos[site_id]
   
   # 更好：索引数组
   positions = data.site_xpos[site_ids]
   ```

---

## 常见问题

### Q: mujoco.MjData 支持多环境吗？
**A:** 原生不支持。你需要创建多个 MjData 对象或使用 DeepMind 的 `dm_control` 库中的向量化 wrapper。

### Q: 如何获取 body 的旋转信息？
**A:** 使用 `data.body_xmat[body_id]` 获取 3x3 旋转矩阵，或使用 `data.body_xquat[body_id]` 获取四元数。

### Q: site 和 body 有什么区别？
**A:** 
- **Body**: 刚体，有质量、惯性等物理属性
- **Site**: 参考点，没有物理属性，用于传感器、接触、约束等

### Q: 数据何时更新？
**A:** 调用 `mujoco.mj_step()` 后，所有数据自动更新。无需手动刷新。

---

## 推荐迁移路径

1. **创建包装类**：为 mujoco 创建类似 mtx 的接口
   ```python
   class MuJoCoEnv:
       def get_position(self, site_name):
           site_id = self.site_ids[site_name]
           return self.data.site_xpos[site_id].copy()
   ```

2. **逐步迁移方法**：一次迁移一个方法

3. **使用适配层**：保持 API 兼容性
   ```python
   # 兼容 mtx 和 mujoco
   def get_position(site, data, backend='mtx'):
       if backend == 'mtx':
           return site.get_position(data)
       else:  # mujoco
           return data.site_xpos[site['id']].copy()
   ```

---

## 参考资源

- [MuJoCo Python 文档](https://mujoco.readthedocs.io/)
- [MuJoCo API 参考](https://mujoco.readthedocs.io/en/latest/APIreference/)
- [dm_control 库](https://github.com/deepmind/dm_control)（包含向量化支持）
