import mujoco
import mujoco.viewer
import os

os.chdir('/Users/qunz/projects_mac/own/rl_learn_zq_native/Task3/o6_urdf/')

try:
    model = mujoco.MjModel.from_xml_path('scene.xml')
    data = mujoco.MjData(model)
    print("🚀 成功！场景已加载，手和方块同框。")
    mujoco.viewer.launch(model, data)
except Exception as e:
    print(f"❌ 依然失败: {e}")