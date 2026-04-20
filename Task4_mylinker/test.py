import mujoco
import numpy as np
import os

# 1. 替换为你的 URDF 或 MJCF 文件路径
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'linkerhand_o6_right.urdf')
try:
    # 加载模型
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print(f"{'ID':<5} | {'Joint Name':<25} | {'Type':<10} | {'Limits (Rad)':<15}")
    print("-" * 60)

    for i in range(model.njnt):
        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        j_type = model.jnt_type[i] # 3 是转动关节 (hinge)
        j_limit = model.jnt_range[i]
        print(f"{i:<5} | {j_name:<25} | {j_type:<10} | {j_limit}")

except Exception as e:
    print(f"加载模型失败: {e}")

"""
(rl_learn) qunz@QundeMacBook-Air rl_learn_zq_native % python Task3/o6_urdf/test.py
ID    | Joint Name                | Type       | Limits (Rad)   
------------------------------------------------------------
0     | rh_thumb_cmc_yaw          | 3          | [0.   1.36]
1     | rh_thumb_cmc_pitch        | 3          | [0.   0.58]
2     | rh_thumb_ip               | 3          | [0.   1.08]
3     | rh_index_mcp_pitch        | 3          | [0.  1.6]
4     | rh_index_dip              | 3          | [0.   1.43]
5     | rh_middle_mcp_pitch       | 3          | [0.  1.6]
6     | rh_middle_dip             | 3          | [0.   1.43]
7     | rh_ring_mcp_pitch         | 3          | [0.  1.6]
8     | rh_ring_dip               | 3          | [0.   1.43]
9     | rh_pinky_mcp_pitch        | 3          | [0.  1.6]
10    | rh_pinky_dip              | 3          | [0.   1.43]
(rl_learn) qunz@QundeMacBook-Air rl_learn_zq_native % 
"""

"""
现实中的控制序号 (SDK)	现实中的功能描述	对应的 MuJoCo 关节名称	关节 ID	运动范围 (弧度)
0	大拇指弯曲	rh_thumb_cmc_pitch	1	0 到 0.58
1	大拇指横摆	rh_thumb_cmc_yaw	0	0 到 1.36
2	食指弯曲	rh_index_mcp_pitch	3	0 到 1.6
3	中指弯曲	rh_middle_mcp_pitch	5	0 到 1.6
4	无名指弯曲	rh_ring_mcp_pitch	7	0 到 1.6
5	小拇指弯曲	rh_pinky_mcp_pitch	9	0 到 1.6
"""