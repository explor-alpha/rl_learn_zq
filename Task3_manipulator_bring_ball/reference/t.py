# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np

_DEFAULT_VALUE_AT_MARGIN = 0.1  # 默认保持 0.1 即可，这已经过 DeepMind 大量实验验证是最优的“边缘衰减值”


def tolerance(
    x: np.ndarray,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = _DEFAULT_VALUE_AT_MARGIN,
) -> np.ndarray:
    """
    tolerance：划分 3 个区域：
    1. Bounds（核心区）：闭区间。得分恒为 1.0。
    2. Margin（缓冲区）：从 Bounds 边界向外辐射 margin 长度的区域。得分依据 _sigmoids 函数，从 1.0 平滑下降至 value_at_margin（0.1）。
    3. Beyond Margin（远端区）：超出 margin 的区域。根据所选 Sigmoid 类型，得分渐进趋于 0（如 long_tail）或硬性截断为 0（如 linear）。

    核心优势：
    1. 将 reward 控制在 [0,1]之间，奖励极度稳定 & 可控。
        reward 分布稳定：避免“[巨大负值，*]的分布破坏梯度的问题“
        可控：各任务奖励上限统一（均为 1.0），可以精确通过 weight 调整不同子任务（如抓取 vs 移动）的影响力对比。
    2. Bounds（核心区）：提供了一个“足够好”的范围。只要进入这个范围，得分就是 1.0。这避免了机器人为了追求“绝对 0 误差”而产生不必要的震颤（Chattering）。
    3. Margin（引导区）：在核心区外提供连续的梯度。机器人只要朝目标挪动一点点，分数就会按曲线提升。
    其中：
        长尾模式 (long_tail)：即使机械臂飞到了天边，离球 2 米远，它依然能提供微弱的梯度（如 0.01 分）。这保证了探索永不掉队，非常适合并行环境下的广域搜索。
        高斯模式 (gaussian)：在靠近目标时提供极佳的平滑度，帮助机器人实现从“粗略靠近”到“精准对齐”的过渡。

    PS： 
    - np.where(condition, [x, y])；
        - 如果 condition (条件) 为真，就取 x，否则取 y。
        - 支持数组 Array
        - Numpy 的“广播机制”和对标量的兼容性，使标量类型也可以兼容
    - “[巨大负值，*]的分布破坏梯度的问题“：
        1. 离群值的梯度屏蔽：
            PPO 的 Value Loss 通常是基于 MSE 的误差回归，理论上应该是在小范围内的微调。方差主要由"正常"的 reward 产生，-100,-10 这种无效 reward 不应该主导 reward 的分布方差。
            换一个角度，mini-batch SGD 方法执行 optimizer.step() 时，如果某个 experience 的 reward 是极小的负值（即误差极大，Loss Function 极大，梯度极大）
            权重的更新方向几乎完全由这个异常数据主导，导致模型过度拟合这个异常数据，忽略了其他正常数据的学习信号。这会导致训练过程不稳定，甚至完全失控。

            尽管在环境外部会嵌套一层（例如VecNormalize基于滑动平均全局缩放）用于归一化 reward，
            但这只能解决“量纲”问题（把 -100 变成 -1），但无法解决“相对显著性”问题。
            离群样本相对于 Batch 内其他样本的相对误差依然巨大，其在梯度下降中的主导地位并没有改变。
        2. 高方差：
            PPO 决定“要不要多做这个动作”的唯一依据为优势函数 A。A 的计算依赖于 R。会继承 R 的高方差。
            若 R 为高方差，智能体就会陷入非稳态更新（在原地左右横跳，训练曲线剧烈抖动），很难静下心来去磨练精准的抓取动作。
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError("lower bound must be less than upper bound")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    in_bounds = np.logical_and(lower <= x, x <= upper)  # 布尔数组；检查 x 是否落在指定的 [lower, upper] 闭区间内
    if margin == 0:   # 没有设置缓冲区；要么 100 分，要么 0 分。这通常用于判别式的任务（比如：球是否掉出了界外）
        value = np.where(in_bounds, 1.0, 0.0)
    else:
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

    return value


def _sigmoids(x, value_at_1, sigmoid):
    if sigmoid in ("cosine", "linear", "quadratic"):
        if not 0 <= value_at_1 < 1:
            raise ValueError(f"`value_at_1` must be nonnegative and smaller than 1, got {value_at_1}.")
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError(f"`value_at_1` must be strictly between 0 and 1, got {value_at_1}.")

    if sigmoid == "gaussian":
        """
        $e^{-x^2}$
        特性：顶部非常平滑。这意味着当机械臂微调位置时，奖励变化很小；但离开目标一定距离后，奖励会迅速滑落。
        “最常用，最标准”；最终的精准对齐
        """
        scale = np.sqrt(-2 * np.log(value_at_1))
        return np.exp(-0.5 * (x * scale) ** 2)

    elif sigmoid == "long_tail":
        """
        $1 / (x^2 + 1)$。
        特性：它的下降速度比高斯慢得多，且保留梯度，永远不会真正到达 0。
        “最适合远距离探索”
        """
        scale = np.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale) ** 2 + 1)

    elif sigmoid == "tanh_squared":
        """
        $1 - \tanh(x)^2$
        特性：这是 tanh 函数的导数形状。它在中间区域的斜率（梯度）非常稳定。
        “物理感最强”；需要平滑力矩控制的任务。
        """
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale) ** 2
    
    elif sigmoid == "hyperbolic":
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)
    
    elif sigmoid == "reciprocal":
        """
        倒数型
        特性：下降极快，但在远端保留极长的尾巴。
        """
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == "linear":
        """
        线性/二次型
        特性：它们是有边界的。一旦超过某个距离，得分会硬性降为 0.0。
        风险：在 RL 中要慎用，因为 0 奖励区域没有梯度，机器人会“迷路”。
        """
        scale = 1 - value_at_1
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == "quadratic":
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    else:
        raise ValueError(f"Unknown sigmoid type {sigmoid!r}.")