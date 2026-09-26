"""平均场解析公式（来自 papers/main.tex 当前版 / knowledge/main_model.md）。

纯函数，输入用实测 <k>_0=k0、<k^2>_0=k2、<b>=b_mean、容量 C。
行号标注取 main.tex（0809 Laplace 框架版）；符号见 knowledge/graph_theory_meta.md。

记号对应 main.tex：
  S_h_1star = S_h^1（大根，级联首次触发点）     S_h_2star = S_h^2（小根，级联区下界）
  S_h_star  = S_h*（B(t)=1 的密度）            S_h_E = 1/<k>_0（B=0，级联终止）

T_c 两分支（Tc_summary）：
  Case I  (ductile, C >= C*): T_c = t^1_L + (t^E - t^1)_MF（零参数）
    初始段用 Laplace 修正（survivor bias），级联段保持平均场 ODE，尾段置零。
  Case II (brittle,  C < C*): T_c = lambda * t^1_L（触发即穿透）
    lambda=1 为零参数单边上界；lambda(N)=a+b/ln N 为论文的校准预测
    （见 lambda_calibration）。
"""
from __future__ import annotations

import math


def collapse_prob_p(N: int, m: int, b_mean: float) -> float:
    """单超边单步坍塌概率 p = (m+1)!/N^m * <b>（main.tex L111）。"""
    return math.factorial(m + 1) / N ** m * b_mean


def branching_B(k0: float, m: int, S_h: float = 1.0) -> float:
    """时变分支数 B(t) = m(<k>_0 * S_h(t) - 1)（L97）。S_h=1 → B(0)。"""
    return m * (k0 * S_h - 1.0)


def rho_c(k0: float, k2: float, m: int) -> float:
    """密度渗流阈值 rho_c = <k>_0 / (m(<k^2>_0 - <k>_0))（L226-228）。"""
    return k0 / (m * (k2 - k0))


def molloy_reed_ratio(k0: float, k2: float) -> float:
    """<k(k-1)>/<k> = (<k^2> - <k>)/<k>。> 1/m 时（初始）有巨分量。"""
    return (k2 - k0) / k0


def has_initial_gc(k0: float, k2: float, m: int) -> bool:
    """初始是否有巨分量：S_h(0)=1 > rho_c，等价 (<k^2>-<k>)/<k> > 1/m。"""
    return molloy_reed_ratio(k0, k2) > 1.0 / m


def S_h_1star(k0: float, m: int, C: float) -> float:
    """级联触发上界 S_h^1（大根，加号，L149-152）。

    方程 C<k>_0 S^2 - (C+<k>_0)S + (m-1)/m = 0 的较大根。
    S_h(t) 从 1 向下演化，首次进入触发区间的点就是 S_h^1。
    判别式恒非负（m>=2），无需复数处理。
    """
    disc = (C + k0) ** 2 - 4.0 * C * k0 * (m - 1) / m
    return (C + k0 + math.sqrt(disc)) / (2.0 * C * k0)


def S_h_2star(k0: float, m: int, C: float) -> float:
    """级联触发下界 S_h^2（小根，减号，L149-152）。"""
    disc = (C + k0) ** 2 - 4.0 * C * k0 * (m - 1) / m
    return (C + k0 - math.sqrt(disc)) / (2.0 * C * k0)


def S_h_star(k0: float, m: int) -> float:
    """B(t)=1 的密度 S_h* = (m+1)/(m<k>_0)（L206）。

    S_h < S_h* ⟺ B(t) < 1；S_h* <= S_h <= S_h^1 ⟺ B(t) >= 1。
    """
    return (m + 1) / (m * k0)


def S_h_E(k0: float) -> float:
    """B(t)=0 的密度 S_h^E = 1/<k>_0（main.tex L155-157）。

    级联需要 B(t)>=0，故级联区为 [S_h^E, S_h^1]；S_h^E -> 0 是级联后的随机坍塌段
    （φ=p）。可证 S_h^2 < S_h^E 恒成立（f(1/k0) = -1/m < 0）。
    """
    return 1.0 / k0


def C_threshold(k0: float, m: int) -> float:
    """两/三阶段判据阈值 C* = 2m/(m+1) * <k>_0（L208/210）。

    C >= C*（等价 S_h^1 <= S_h*）→ 两阶段；C < C*（等价 S_h^2 < S_h* < S_h^1）→ 三阶段。
    """
    return 2.0 * m / (m + 1) * k0


def regime(k0: float, m: int, C: float) -> str:
    """按容量判 regime：'case_I'（C>=C*，ductile）或 'case_II'（C<C*，brittle）。

    注意：Case I/II 与阶段数不等价——Case I 三阶段（初始 + 0<=B<1 级联 + 随机段），
    Case II 四阶段（初始 + B>=1 爆炸 + 0<=B<1 级联 + 随机段，实际被一步穿透跳过）。
    """
    return "case_I" if C >= C_threshold(k0, m) else "case_II"


def C_crit(k0: float, k2: float, m: int) -> float:
    """临界容量 C_crit：由 rho_c = S_h^2 解出。

    rho_c > S_h^2 ⟺ C > C_crit（S_h^2 随 C 单调递减；v6 修正方向）。
    0809 版论文已移除 C_crit 约束（T_c 公式不引用 S_h^2/C_crit）；本函数保留作
    诊断量（run_step1 报告用，历史见 findings ⑭）。
    分母为 0（Molloy-Reed 临界，rho_c=1）时返回 inf。
    """
    denom = k0 * (k0 ** 2 - m * (k2 - k0))
    if abs(denom) < 1e-300:
        return float("inf")
    return m * (k2 - k0) * (k0 ** 2 + (1 - m) * (k2 - k0)) / denom


def t_1(N: int, m: int, b_mean: float, S_h_1_val: float) -> float:
    """平均场初始段触发时刻 t^1_MF = (1/p) ln(1/S_h^1)（L287）。"""
    p = collapse_prob_p(N, m, b_mean)
    if p == 0.0:
        raise ValueError("p=0 (b_mean=0)：永不坍塌，T_c 无定义")
    return (1.0 / p) * math.log(1.0 / S_h_1_val)


def laplace_transform_b(s: float, b_dist: str = "uniform",
                        b_param: float = 1.0) -> float:
    """脆弱度分布的 Laplace 变换 B̂(s) = E[e^{-s b}]（main.tex L256-259）。

      "uniform"  b ~ U(0, b_param):  (1 - e^{-s*beta}) / (s*beta)
      "constant" b ≡ b_param:        e^{-s*b0}
    """
    if b_dist == "uniform":
        x = s * b_param
        return 1.0 if x <= 0.0 else (1.0 - math.exp(-x)) / x
    if b_dist == "constant":
        return math.exp(-s * b_param)
    raise ValueError(f"unsupported b_dist: {b_dist!r}")


def b_mean_of(b_dist: str = "uniform", b_param: float = 1.0) -> float:
    """该脆弱度分布的 <b>（uniform 为 beta/2，constant 为 b0）。"""
    if b_dist == "uniform":
        return 0.5 * b_param
    if b_dist == "constant":
        return b_param
    raise ValueError(f"unsupported b_dist: {b_dist!r}")


def S_h_laplace(t: float, N: int, m: int, b_dist: str = "uniform",
                b_param: float = 1.0) -> float:
    """Laplace 修正的初始段密度（L261-263）：S_h(t) = [B̂(ct)]^{m+1}，c = m!/N^m。

    b~U(0,1) 即论文闭式 ((1-e^{-ct})/(ct))^{m+1}；e^{-pt} 是它的一阶近似，
    survivor bias（高危边先死）使真实衰减更慢。常数 b 无异质性，此式退化为 e^{-pt}。
    """
    c = math.factorial(m) / N ** m
    return laplace_transform_b(c * t, b_dist, b_param) ** (m + 1)


def t_1_laplace(N: int, m: int, S_h_1_val: float, b_dist: str = "uniform",
                b_param: float = 1.0) -> float:
    """Laplace 修正触发时刻 t^1_L：解 S_h_laplace(t) = S_h^1（L264-269）。

    单调递减函数上的二分求根（论文对 b~U(0,1) 另给 Lambert W 显式解，数值上等价）。
    恒有 t^1_L >= t^1_MF（Jensen：E[e^{-p_e t}] >= e^{-pt}），常数 b 时取等号。
    """
    if not (0.0 < S_h_1_val < 1.0):
        raise ValueError(f"S_h^1={S_h_1_val} 不在 (0,1)：退化容量区（S_h^1>=1 ⟺ "
                         "C <= (k0-(m-1)/m)/(k0-1)），t^1_L 无定义")
    c = math.factorial(m) / N ** m
    lo, hi = 0.0, 1.0 / c
    while S_h_laplace(hi, N, m, b_dist, b_param) > S_h_1_val:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if S_h_laplace(mid, N, m, b_dist, b_param) > S_h_1_val:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def D_propagation(N: int, m: int) -> float:
    """级联传播步数 D = ln N / ln m（L202-204，用实测 |E|=N<k>/(m+1)）。"""
    return math.log(N) / math.log(m)


def t_E(N: int, m: int, k0: float, b_mean: float, C: float) -> float:
    """平均场级联终止时刻 t^E_MF（L288-292，Case I 级联段 ODE 积分到 S_h^E）：

      t^E = t^1 + (m<k>_0/p)(1/<k>_0 - S_h^1) + ((m+1)/p) ln(S_h^1 <k>_0)
    由 dS_h/dt = -p S_h/(1-B) 从 S_h^1 积分到 S_h^E = 1/<k>_0。
    """
    sh1 = S_h_1star(k0, m, C)
    t1 = t_1(N, m, b_mean, sh1)
    p = collapse_prob_p(N, m, b_mean)
    shE = S_h_E(k0)
    return t1 + (m * k0 / p) * (shE - sh1) + (m + 1) / p * math.log(sh1 * k0)


def collapse_time_Tc_case_I(N: int, m: int, k0: float, k2: float,
                            b_mean: float, C: float,
                            b_dist: str = "uniform", b_param: float = 1.0) -> float:
    """Case I（ductile）坍塌时间 T_c（C >= C*，main.tex L295-298）：

      T_c^(I) = t^1_L + (t^E - t^1)_MF
    初始段用 Laplace 修正（survivor bias），级联段（S_h^1 -> S_h^E）保持平均场
    ODE 积分，级联后尾段置零（多数 seed 尾段时长恰为 0，findings ⑬；N=1e4 时
    8/15 seed 尾段非零但均值仅占 T_c 1.5%）。
    零参数；数值验证误差 -0.3/-0.3/+2.9/+9.1/+8.9%（N=500/1000/2000/5000/1e4，
    15 seed 均值，findings ⑰）——±5% 仅在 N<=2000 成立，大 N 处 ~+9%
    （级联段 sim/MF 降至 0.27-0.33 + 触发略提前）。

    k2 不进入公式，保留于签名以统一接口；有效性条件 rho_c(k0,k2,m)*k0 < 1 需 k2。
    """
    sh1 = S_h_1star(k0, m, C)
    t1L = t_1_laplace(N, m, sh1, b_dist, b_param)
    casc = t_E(N, m, k0, b_mean, C) - t_1(N, m, b_mean, sh1)
    return t1L + casc


def lambda_calibration(N: int, a: float = 0.583, b: float = 1.808) -> float:
    """Case II 校准因子 lambda(N) = a + b/ln N（饱和式，main.tex Case II 校准式）。

    经验校准：t^1_L 是单边上界（波动使触发提前，extreme-value 效应），lambda(N)
    把它压回预测值。物理动机是极值启发（最大负载随 |E| ∝ N 增长），函数形式为
    经验拟合。

    默认 a=0.583/b=1.808 是 8 个 N（500-1e5，各 15 seed）的 sem 加权最小二乘系数
    （精确 LSQ 值 0.5833/1.8083）：加权 chi^2=6.1（dof 6），逐点偏差
    +0.5/-2.2/-0.6/+1.2/-0.2/-0.6/+0.3/-0.3%（最大 2.2%，在 N=1000），
    **全 200 倍 N 跨度可用，不需要再限定适用域**。旧对数式 1-gamma*ln(N/N0) 同为 2 参数但 chi^2=14.7，
    且按 N<=1e4 标定后外推到 1e5 偏 -6.6%，已弃用（findings ㉓㉕）。

    外推含义：N->inf 时 lambda -> a=0.583（有限平台），对数式则断言 lambda->0
    （非物理）。平台值本身是函数形式带来的断言，不是实测结论——实测区间只覆盖
    到 lambda≈0.74。

    N<=1 时 ln N<=0 无定义，抛 ValueError。
    """
    if N <= 1:
        raise ValueError(f"lambda(N={N})：校准式要求 N>1（ln N>0）")
    return a + b / math.log(N)


def collapse_time_Tc_case_II(N: int, m: int, k0: float, k2: float,
                             b_mean: float, C: float,
                             lam: float = 1.0,
                             b_dist: str = "uniform", b_param: float = 1.0) -> float:
    """Case II（brittle）坍塌时间 T_c（C < C*，main.tex Tc_II_repeat/Tc_summary）：

      T_c^(II) = lambda * t^1_L（触发即穿透，级联传播时间可忽略）
    负载裕量小（Q ≲ ΔL），触发事件一步移除宏观比例的边并穿透 rho_c。
    lam=1（默认）为零参数单边上界：120/120 seed 理论 > 仿真，误差
    +15.0/+15.7/+21.1/+27.2/+28.1/+29.8/+33.7/+34.7%（N=500…1e5，15 seed 均值）
    随 N 增大、增速放缓。lam=lambda_calibration(N) 为论文的校准预测
    （饱和式 a=0.583/b=1.808，8 点加权 LSQ）：与 λ 观测
    （0.870/0.864/0.826/0.786/0.781/0.770/0.748/0.742）偏差 <=2.2%（findings ㉕）。

    b_mean、k2 不进入公式（脆弱度只通过 Laplace 变换 B̂ 进入），保留于签名以统一接口。
    """
    sh1 = S_h_1star(k0, m, C)
    return lam * t_1_laplace(N, m, sh1, b_dist, b_param)


def collapse_time_Tc(N: int, m: int, k0: float, k2: float,
                     b_mean: float, C: float,
                     b_dist: str = "uniform", b_param: float = 1.0) -> float:
    """按 C 自动分派 Case I/II 的 T_c（main.tex L308-325）。"""
    if regime(k0, m, C) == "case_I":
        return collapse_time_Tc_case_I(N, m, k0, k2, b_mean, C, b_dist, b_param)
    return collapse_time_Tc_case_II(N, m, k0, k2, b_mean, C,
                                    b_dist=b_dist, b_param=b_param)
