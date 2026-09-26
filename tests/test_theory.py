"""theory.py 单元测试（main.tex 0809 Laplace 框架版公式）。"""
import math

import pytest
from scipy.integrate import quad
from scipy.special import lambertw

from src import theory


def test_collapse_prob_p():
    # p = (m+1)!/N^m * <b> = 6/1000^2 * 0.5
    assert theory.collapse_prob_p(1000, 2, 0.5) == pytest.approx(6 / 1000**2 * 0.5)


def test_branching_B_time_varying():
    # B(0) = m(<k>_0 - 1)
    assert theory.branching_B(2.0, 2, S_h=1.0) == pytest.approx(2.0)   # 2*(2-1)
    # B(t) at S_h=0.5: 2*(2*0.5 - 1) = 0
    assert theory.branching_B(2.0, 2, S_h=0.5) == pytest.approx(0.0)
    # B=1 点 S_h*=(m+1)/(m<k>_0)：m=2,k0=3 → S_h=0.5 → B=1
    assert theory.branching_B(3.0, 2, S_h=0.5) == pytest.approx(1.0)


def test_rho_c():
    # rho_c = <k>_0 / (m(<k^2>_0 - <k>_0)) = 2/(2*(10-2)) = 0.125
    assert theory.rho_c(2.0, 10.0, 2) == pytest.approx(0.125)


def test_molloy_reed_and_initial_gc():
    assert theory.molloy_reed_ratio(2.0, 10.0) == pytest.approx(4.0)  # (10-2)/2
    assert theory.has_initial_gc(2.0, 10.0, 2) is True          # 4 > 0.5
    assert theory.has_initial_gc(2.0, 2.5, 2) is False           # 0.25 < 0.5


def test_S_h_roots_at_C_star():
    # m=2, k0=3: C*=4 处判别式=25，两根 = 1/12（小根）与 1/2（大根=S_h*）
    k0, m, C = 3.0, 2, 4.0
    assert theory.S_h_2star(k0, m, C) == pytest.approx(1 / 12)
    assert theory.S_h_1star(k0, m, C) == pytest.approx(0.5)
    assert theory.S_h_1star(k0, m, C) == pytest.approx(theory.S_h_star(k0, m))
    # 大根 > 小根，且都在 (0,1)
    assert 0 < theory.S_h_2star(k0, m, C) < theory.S_h_1star(k0, m, C) < 1


def test_S_h_1star_above_S_h_star_for_case_II():
    # C < C*（Case II）：S_h^2 < S_h* < S_h^1
    k0, m, C = 3.0, 2, 2.0
    s1, s2 = theory.S_h_1star(k0, m, C), theory.S_h_2star(k0, m, C)
    ss = theory.S_h_star(k0, m)
    assert s2 < ss < s1


def test_S_h_star_zero_triggers_at_m1():
    # m=1: (m-1)/m=0 → 判别式=(C+k0)^2 → 小根=0（级联区退化）
    assert theory.S_h_2star(3.0, 1, 2.0) == pytest.approx(0.0)


def test_degenerate_capacity_threshold():
    # S_h^1 >= 1 ⟺ C <= (k0-(m-1)/m)/(k0-1)（main.tex L160 排除的退化容量区；
    # 注意 tex 原文把不等号方向写反，正确方向以此测试为准）
    k0, m = 6.5, 2
    C_th = (k0 - (m - 1) / m) / (k0 - 1)          # = 6.0/5.5 ≈ 1.0909
    assert theory.S_h_1star(k0, m, C_th) == pytest.approx(1.0)
    assert theory.S_h_1star(k0, m, C_th - 0.04) > 1.0     # C 低于阈值 → 退化
    assert theory.S_h_1star(k0, m, C_th + 0.04) < 1.0     # C 高于阈值 → 正常


def test_C_threshold_and_regime_direction():
    # C* = 2m/(m+1)<k>_0 = 8/3（m=2,k0=3 → 4）
    assert theory.C_threshold(3.0, 2) == pytest.approx(4.0)
    assert theory.regime(3.0, 2, 2.0) == "case_II"        # C < C*：brittle
    assert theory.regime(3.0, 2, 8.0) == "case_I"         # C > C*：ductile
    assert theory.regime(3.0, 2, 4.0) == "case_I"         # 边界 C=C* 归 Case I


def test_C_crit_matches_direct_solution():
    # C_crit（诊断量，0809 论文已移除该约束）应等于从 S_h^2(C)=rho_c 解出的 C：
    #   C = (k0*rho - (m-1)/m) / (rho*(k0*rho - 1))
    for k0, k2, m in [(3.0, 20.0, 2), (5.0, 50.0, 3), (3.0, 15.0, 5)]:
        rc = theory.rho_c(k0, k2, m)
        C_direct = (k0 * rc - (m - 1) / m) / (rc * (k0 * rc - 1))
        assert theory.C_crit(k0, k2, m) == pytest.approx(C_direct, rel=1e-10)
    # 大 ⟨k²⟩₀（幂律）下 C_crit 很大：2·297·(9-297)/(3·(9-594)) = 171072/1755 ≈ 97.477
    assert theory.C_crit(3.0, 300.0, 2) == pytest.approx(171072 / 1755, rel=1e-10)


def test_t_1():
    # t^1_MF = (1/p) ln(1/S_h^1)
    N, m, k0, C, b = 1000, 2, 3.0, 2.0, 0.5
    p = theory.collapse_prob_p(N, m, b)
    s1 = theory.S_h_1star(k0, m, C)
    assert theory.t_1(N, m, b, s1) == pytest.approx(math.log(1 / s1) / p)


def test_S_h_laplace_properties():
    # t=0 → 1；单调递减；Jensen：E[e^{-p_e t}] >= e^{-pt}（survivor bias 衰减更慢）
    N, m = 1000, 2
    p = theory.collapse_prob_p(N, m, 0.5)      # U(0,1) 的 <b>=0.5
    assert theory.S_h_laplace(0.0, N, m) == pytest.approx(1.0)
    prev = 1.0
    for t in (1e4, 1e5, 5e5, 2e6):
        v = theory.S_h_laplace(t, N, m)
        assert 0.0 < v < prev
        assert v > math.exp(-p * t)
        prev = v


def test_t_1_laplace_root_and_survivor_bias():
    # 根代回残差为 0；且 t^1_L > t^1_MF（b̄=0.5 基准）
    for N, m, k0, C in [(500, 2, 6.0, 11.94), (1000, 2, 6.5, 13.074),
                        (2000, 3, 5.0, 8.0)]:
        sh1 = theory.S_h_1star(k0, m, C)
        t1L = theory.t_1_laplace(N, m, sh1)
        assert theory.S_h_laplace(t1L, N, m) == pytest.approx(sh1, rel=1e-9)
        assert t1L > theory.t_1(N, m, 0.5, sh1)


def test_t_1_laplace_matches_lambert_w():
    # 论文 L265-268 的 Lambert W 显式解与二分求根一致：
    #   t^1_L = (1/c)[W_0(-e^{-1/α}/α) + 1/α],  α = (S_h^1)^{1/(m+1)}
    for N, m, k0, C in [(1000, 2, 6.5, 13.074), (500, 2, 6.0, 5.97)]:
        sh1 = theory.S_h_1star(k0, m, C)
        alpha = sh1 ** (1.0 / (m + 1))
        c = math.factorial(m) / N ** m
        t_lw = (lambertw(-math.exp(-1.0 / alpha) / alpha, 0).real + 1.0 / alpha) / c
        assert theory.t_1_laplace(N, m, sh1) == pytest.approx(t_lw, rel=1e-9)


def test_case_I_Tc_matches_formula():
    # Case I（C > C*）T_c = t^1_L + (t^E - t^1)_MF（L295-298）逐项核对
    N, m, k0, k2, b, C = 1000, 2, 3.0, 20.0, 0.5, 8.0
    sh1 = theory.S_h_1star(k0, m, C)
    t1L = theory.t_1_laplace(N, m, sh1)
    t1 = theory.t_1(N, m, b, sh1)
    tE = theory.t_E(N, m, k0, b, C)
    p = theory.collapse_prob_p(N, m, b)
    shE = theory.S_h_E(k0)
    assert tE == pytest.approx(
        t1 + (m * k0 / p) * (shE - sh1) + (m + 1) / p * math.log(sh1 * k0))
    assert theory.collapse_time_Tc_case_I(N, m, k0, k2, b, C) == pytest.approx(
        t1L + (tE - t1))


def test_case_II_Tc_is_t1_laplace():
    # Case II（C < C*）脆性极限 T_c = t^1_L（L303-305，零参数、单边上界）
    N, m, k0, k2, b, C = 1000, 2, 3.0, 20.0, 0.5, 2.0
    sh1 = theory.S_h_1star(k0, m, C)
    Tc = theory.collapse_time_Tc_case_II(N, m, k0, k2, b, C)
    assert Tc == pytest.approx(theory.t_1_laplace(N, m, sh1))
    # 上界结构：Laplace 修正必然晚于平均场 t^1
    assert Tc > theory.t_1(N, m, b, sh1)


def test_lambda_calibration():
    # lambda(N) = a + b/ln N（main.tex Case II 校准式，8 点加权 LSQ
    # a=0.583, b=1.808）
    assert theory.lambda_calibration(5000) == pytest.approx(
        0.583 + 1.808 / math.log(5000.0))
    # N 单调递减，且始终为正（对数式会跨零，饱和式不会）
    assert (theory.lambda_calibration(1000)
            > theory.lambda_calibration(2000)
            > theory.lambda_calibration(10000)
            > theory.lambda_calibration(int(1e15))
            > 0.583)
    # 逐点复现八个 N 的实测 λ（15 seed 均值）：论文取的两位/三位有效系数下最大
    # 偏差 2.23%（N=1000），精确 LSQ 系数 0.5833/1.8083 下为 2.20%
    observed = {500: 0.8696, 1000: 0.8640, 2000: 0.8261, 5000: 0.7863,
                10_000: 0.7809, 20_000: 0.7702, 50_000: 0.7477, 100_000: 0.7422}
    for N, lam in observed.items():
        assert abs(theory.lambda_calibration(N) / lam - 1.0) <= 0.023
    # 定义域守护：ln N <= 0
    with pytest.raises(ValueError):
        theory.lambda_calibration(1)


def test_case_II_lam_scales_t1_laplace():
    # T_c^(II) = lam * t^1_L：lam 纯乘子；默认 lam=1 为零参数上界
    N, m, k0, k2, b, C = 1000, 2, 3.0, 20.0, 0.5, 2.0
    base = theory.collapse_time_Tc_case_II(N, m, k0, k2, b, C)
    lam = theory.lambda_calibration(N)
    assert theory.collapse_time_Tc_case_II(N, m, k0, k2, b, C, lam=lam) \
        == pytest.approx(lam * base)


def test_case_I_Tc_matches_numerical_integral():
    """Case I T_c 与直接数值积分对照：

    T_c = t^1_L + (1/p)∫_{S_h^E}^{S_h^1} (1-B)/s ds  [级联段 0<=B<1，尾段置零]
    """
    for m, N, k0, k2, C in [(2, 1000, 3.0, 20.0, 8.0), (3, 2000, 3.0, 30.0, 6.0),
                            (2, 500, 1.85, 9.0, 3.0)]:
        b = 0.5
        p = theory.collapse_prob_p(N, m, b)
        sh1 = theory.S_h_1star(k0, m, C)
        shE = theory.S_h_E(k0)
        t1L = theory.t_1_laplace(N, m, sh1)

        def integrand(s):
            B = m * (k0 * s - 1.0)
            return (1.0 - B) / s

        seg, _ = quad(integrand, shE, sh1, limit=500)   # 级联段到 S_h^E
        num = t1L + seg / p
        th_ = theory.collapse_time_Tc_case_I(N, m, k0, k2, b, C)
        assert th_ == pytest.approx(num, rel=1e-9)
        # 物理：T_c 必须 > t^1_L（级联段时长为正）
        assert th_ > t1L


def test_Tc_positive_for_realistic_params():
    """真实参数（小 p）下 T_c 必须为正——回归守护：ln 项漏 1/p 会给出负时间。

    Case I 要求 T_c > t^1_L；Case II（脆性）T_c = t^1_L。
    """
    for m, N, k0, k2, C in [(2, 200, 1.85, 9.0, 8.0), (2, 200, 1.85, 9.0, 2.0),
                            (3, 1000, 3.0, 20.0, 4.0), (3, 1000, 3.0, 20.0, 2.0)]:
        b = 0.5
        Tc = theory.collapse_time_Tc(N, m, k0, k2, b, C)
        assert Tc > 0
        sh1 = theory.S_h_1star(k0, m, C)
        t1L = theory.t_1_laplace(N, m, sh1)
        if theory.regime(k0, m, C) == "case_I":
            assert Tc > t1L
        else:
            assert Tc == pytest.approx(t1L)


def test_dispatch_by_regime():
    N, m, k0, k2, b = 1000, 2, 3.0, 20.0, 0.5
    assert theory.collapse_time_Tc(N, m, k0, k2, b, 8.0) == pytest.approx(
        theory.collapse_time_Tc_case_I(N, m, k0, k2, b, 8.0))
    assert theory.collapse_time_Tc(N, m, k0, k2, b, 2.0) == pytest.approx(
        theory.collapse_time_Tc_case_II(N, m, k0, k2, b, 2.0))


def test_degenerate_guards():
    # p=0（b_mean=0）→ 永不坍塌
    with pytest.raises(ValueError):
        theory.t_1(1000, 2, 0.0, 0.5)
    # S_h^1 >= 1（退化容量区）→ t^1_L 无定义
    with pytest.raises(ValueError):
        theory.t_1_laplace(1000, 2, 1.0)
    with pytest.raises(ValueError):
        theory.t_1_laplace(1000, 2, 1.2)


# --------------------------------------------------------------------------
# 脆弱度分布的推广（<b> 扫描 / real-data 协议用常数 b）
# --------------------------------------------------------------------------

def test_laplace_transform_uniform_matches_paper_closed_form():
    """b~U(0,1) 时 B̂(s) 必须回到论文闭式 (1-e^{-s})/s。"""
    for s in (1e-6, 0.1, 1.0, 5.0):
        assert theory.laplace_transform_b(s) == pytest.approx((1 - math.exp(-s)) / s)
    assert theory.laplace_transform_b(0.0) == pytest.approx(1.0)


def test_laplace_transform_constant_is_exponential():
    for b0 in (0.2, 0.4, 1.0):
        for s in (0.1, 1.0, 3.0):
            assert theory.laplace_transform_b(s, "constant", b0) == pytest.approx(
                math.exp(-s * b0))


def test_b_mean_of():
    assert theory.b_mean_of("uniform", 1.0) == pytest.approx(0.5)
    assert theory.b_mean_of("uniform", 0.6) == pytest.approx(0.3)
    assert theory.b_mean_of("constant", 0.4) == pytest.approx(0.4)


def test_constant_b_removes_survivor_bias():
    """常数 b 无异质性 → Laplace 修正退化为平均场：S_h=e^{-pt} 且 t^1_L=t^1_MF。"""
    N, m, b0 = 1000, 2, 0.4
    p = theory.collapse_prob_p(N, m, b0)
    for t in (1e4, 1e5, 5e5):
        assert theory.S_h_laplace(t, N, m, "constant", b0) == pytest.approx(
            math.exp(-p * t))
    sh1 = theory.S_h_1star(6.5, m, 13.0)
    assert theory.t_1_laplace(N, m, sh1, "constant", b0) == pytest.approx(
        theory.t_1(N, m, b0, sh1), rel=1e-6)


@pytest.mark.parametrize("beta", [0.25, 0.5, 1.0])
def test_Tc_scales_inversely_with_b_scale(beta):
    """b~U(0,beta) 只是把所有 p_e 同比例缩放 → T_c 精确正比于 1/<b>。"""
    N, m, k0, k2 = 2000, 2, 6.6, 120.0
    for C in (6.7, 13.4):                       # Case II / Case I
        ref = theory.collapse_time_Tc(N, m, k0, k2, 0.5, C, "uniform", 1.0)
        got = theory.collapse_time_Tc(N, m, k0, k2, 0.5 * beta, C, "uniform", beta)
        assert got == pytest.approx(ref / beta, rel=1e-6)
