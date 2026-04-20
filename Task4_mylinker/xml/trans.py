import mujoco
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(current_dir, "linkerhand_o6_right.urdf")

# 2. 加载 URDF (这一步就是“编译”过程的内核)
model = mujoco.MjModel.from_xml_path(urdf_path)

# 3. 将这个内部模型保存为原生的 MJCF XML 格式
mujoco.mj_saveLastXML('hand_clean.xml', model)

print("✅ 转换成功！")