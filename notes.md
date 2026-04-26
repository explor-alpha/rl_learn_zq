# v2.1 exp 01-05

```python
# env.py reward
        # 计算欧氏距离：结果是数学纯量（Scalar）
        dist_h2b = np.linalg.norm(self.data.site_xpos[self.pinch_id] - self.data.xpos[self.ball_id])
        dist_b2t = np.linalg.norm(self.data.xpos[self.ball_id] - self.data.site_xpos[self.target_id])

        # reward1(-)：引导手靠近球
        reward_reach = - self.w_1 * dist_h2b 

        # reward2(+): 引导抓取：如果手离球很近，给一个额外的“鼓励抓取”奖励
        # 可以通过判断两个手指头的位移，或者简单的距离阈值
        reward_grasp = 0
        if dist_h2b < 0.05:
            # 引导 grasp 关节闭合 (假设 action[4] 是抓取)
            reward_grasp = self.w_2 * (1.0 - dist_h2b/0.05)  # 0.5

        # reward3(+): 带球奖励：只有当球离开地面，或者球与手距离极近时，才增加 dist_b2t 的权重
        # 否则 Agent 会在还没碰到球时就想去 target，导致姿态扭曲
        reward_bring = 0
        if dist_h2b < 0.03:
            reward_bring = 2.0 - (self.w_3 * dist_b2t)  # 2

        reward = reward_reach + reward_grasp + reward_bring

        """
        # 绕路引导
        if obs[8] < 0.2 and obs[10] > 0.2:
            dist_to_gate = np.linalg.norm(self.data.site_xpos[self.pinch_id] - [0.2, 0, self.wall_height + 0.1])
            reward -= self.w_gate * dist_to_gate
        """

        # 终止逻辑 & reward_success
        self.current_step += 1
        terminated = False
        is_success=0
        if dist_b2t < self.success_threshold:
            reward += self.w_success # 给予一笔“终点奖金”
            terminated = True
            is_success = 1.0
```

```python
# config.py
    # 奖励权重
    reward_weight_1 = 1.0 
    reward_weight_2 = 5              # exp04:0.25       # 2       # 5          # exp01:0.5
    reward_weight_3 = 10#env:10      # exp04:4        # 4       # 10.0        # exp01:2
    reward_weight_success = 50.0                    # exp02-05:50   # exp01:10
```

