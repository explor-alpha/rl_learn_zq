# mjpython Task3/o6_urdf/test_visualize.py 

import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path('Task3/o6_urdf/linkerhand_o6_right.urdf')
data = mujoco.MjData(model)

# 映射表中的 6 个主关节 ID
control_ids = [1, 0, 3, 5, 7, 9] 
# 名字对应，方便你观察终端输出
names = ["大拇指弯曲", "大拇指横摆", "食指弯曲", "中指弯曲", "无名指弯曲", "小拇指弯曲"]

with mujoco.viewer.launch_passive(model, data) as viewer:
    joint_idx = 0
    while viewer.is_running():
        step_start = time.time()

        # 1. 所有的关节先复位到 0
        data.qpos[:] = 0 
        
        # 2. 轮流让其中一个关节弯曲到最大值的一半
        target_joint_id = control_ids[joint_idx]
        limit_max = model.jnt_range[target_joint_id][1]
        data.qpos[target_joint_id] = limit_max * 0.8 # 弯曲 80%
        
        if int(time.time()) % 2 == 0: # 每 2 秒换一个手指
            joint_idx = (joint_idx + 1) % 6
            print(f"当前正在测试关节: {names[joint_idx]} (ID: {target_joint_id})")

        mujoco.mj_step(model, data)
        viewer.sync()

        time.sleep(0.1)