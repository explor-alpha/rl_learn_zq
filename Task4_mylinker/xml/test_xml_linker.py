import mujoco
import mujoco.viewer
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(current_dir, "linker.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)
