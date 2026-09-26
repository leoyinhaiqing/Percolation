"""仿真参数配置。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """单次仿真的全部参数。

    默认对应当前框架：m=2、幂律超度 gamma=2.5、脆弱度 U(0,1)。
    容量 C 需显式给（实验取两个值跨过 C* = 2m/(m+1)<k>_0）。
    """

    N: int = 1000                # 节点数
    m: int = 2                   # 阶（每超边 m+1 个节点）
    gamma: float = 2.5           # 幂律超度指数 P(k) ∝ k^{-gamma}
    k_min: int = 1               # 超度下截断
    k_max: Optional[int] = None  # 超度上截断；None → floor(N^{1/(gamma-1)})
    C: Optional[float] = None    # 统一超边容量；实验取两个值跨 C*
    b_dist: str = "uniform"      # 脆弱度分布："uniform"（U(0,b_param)）或 "constant"
    b_param: float = 1.0         # uniform 的上界 beta / constant 的取值 b0
    eps: float = 0.1             # 主巨分量阈值（定义 T_c）
    eps2: float = 0.05           # 次巨分量阈值
    L0: float = 1.0              # 初始负载
    seed: int = 0                # 随机种子（可复现）
    t_max: Optional[int] = None  # 最大步数；None → 自动取 ~K*T_c_th
    record_every: int = 1        # 每隔多少步记录一次指标（含巨分量）

    def __post_init__(self) -> None:
        if self.m < 1:
            raise ValueError("m must be >= 1")
        if self.gamma <= 1.0:
            raise ValueError("gamma must be > 1")
        if self.k_min < 1:
            raise ValueError("k_min must be >= 1")
        if self.k_max is not None and self.k_max < self.k_min:
            raise ValueError("k_max must be >= k_min")
        if self.C is not None and not (self.C > 0.0):
            raise ValueError("C must be > 0 when set")
        if self.b_dist not in ("uniform", "constant"):
            raise ValueError(f"unsupported b_dist: {self.b_dist!r}")
        if not (0.0 < self.b_param <= 1.0):
            raise ValueError("b_param must be in (0, 1]")
