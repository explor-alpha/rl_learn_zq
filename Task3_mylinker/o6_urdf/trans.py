import mujoco
import os

# 1. 确保在正确的目录下
dir_path = '/Users/qunz/projects_mac/own/rl_learn_zq_native/Task3/o6_urdf/'
os.chdir(dir_path)

# 2. 加载 URDF (这一步就是“编译”过程的内核)
# 它会将 URDF 转换为 MuJoCo 内部的 MjModel 结构
model = mujoco.MjModel.from_xml_path('linkerhand_o6_right.urdf')

# 3. 将这个内部模型保存为原生的 MJCF XML 格式
# 这会生成一个极其干净、不包含任何非物理标签的 XML
mujoco.mj_saveLastXML('hand_clean.xml', model)

print("✅ 转换成功！生成的 hand_clean.xml 已经可以被 scene.xml 完美包含。")