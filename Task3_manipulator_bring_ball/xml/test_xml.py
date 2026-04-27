"""
test_xml.py: 测试 XML; 并获取 env.py-render方法 的自由视角相机的默认初始位置参数 lookat, distance, azimuth, elevation
    调用方式: 
        mjpython Task3_manipulator_bring_ball/xml/test_xml.py  
        将结果复制到 env.py 中的相机参数设置里
"""

import mujoco
import mujoco.viewer
import time
import os

# 1. 路径处理
current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "manipulator_bring_ball.xml")

# 2. 加载模型
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 3. 启动可视化窗口
with mujoco.viewer.launch_passive(model, data) as viewer:

    # ------------------------------    
    # --- 通过本脚本自动获取自由视角相机的默认初始位置 ---
    # lookat: 相机盯着看的目标点 (x, y, z)。我们盯着墙和球中间的位置(0.1, 0, 0.2)
    # distance: 相机距离观察点的距离。值越大，画面越小（拉远视角）
    # azimuth: 方位角。90度表示从正前方（沿Y轴）观察 X-Z 平面
    # elevation: 仰角。负值表示向下俯视。-15 到 -20 度比较适合观察立体感
    viewer.cam.lookat = [0.070, 0.009, 0.492]
    viewer.cam.distance = 2.761
    viewer.cam.azimuth = 96.373
    viewer.cam.elevation = -3.769
    # （可选）将结果复制到 env.py 中的相机参数设置里
    last_print_time = time.time()
    # ------------------------------    

    while viewer.is_running():
        step_start = time.time()

        # 物理仿真步进
        mujoco.mj_step(model, data)

        # 刷新渲染
        viewer.sync()

        # ------------------------------
        # --- 每隔 2 秒打印一次相机参数 ---
        # 你可以一边用鼠标调整视角，一边看终端里打印出的数值
        if time.time() - last_print_time > 2.0:
            print("\n当前相机参数 (Copy these to your code):")
            print(f"viewer.cam.lookat = [{viewer.cam.lookat[0]:.3f}, {viewer.cam.lookat[1]:.3f}, {viewer.cam.lookat[2]:.3f}]")
            print(f"viewer.cam.distance = {viewer.cam.distance:.3f}")
            print(f"viewer.cam.azimuth = {viewer.cam.azimuth:.3f}")
            print(f"viewer.cam.elevation = {viewer.cam.elevation:.3f}")
            last_print_time = time.time()
        # ------------------------------

        # 控制仿真速度，使其与物理时间步同步
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)