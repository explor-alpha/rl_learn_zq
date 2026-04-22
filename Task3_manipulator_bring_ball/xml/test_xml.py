# mjpython Task3_manipulator_bring_ball/xml/test_xml.py  

import mujoco
import mujoco.viewer
import time
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "manipulator_bring_ball.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 启动可视化窗口
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # 物理仿真步进
        mujoco.mj_step(model, data)

        # 刷新渲染
        viewer.sync()

        # 控制仿真速度
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)