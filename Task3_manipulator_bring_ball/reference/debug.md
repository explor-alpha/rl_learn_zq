env.py
1. self.data.ctrl[4]是意图不是结果，不能用于判断是否成功抓取
2. 以下获取 h2b 距离的定义模糊：
        hand_pos = self.data.site_xpos[self.pinch_id][[0, 2]]  # 提取 X 和 Z 坐标
        ball_pos = self.data.xpos[self.ball_body_id][[0, 2]]  # 提取 X 和 Z 坐标
        dist_h2b = np.linalg.norm(hand_pos - ball_pos)
3. 