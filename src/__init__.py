"""高阶网络动态负载重分配级联坍塌模型 —— 仿真与理论。

模块：
- config:     参数配置
- theory:     平均场解析公式（B, L_c, p, t*, rho_c, T_c）
- hypergraph: 超图生成、关联结构、超度矩、巨分量
- cascade:    Stage 1 随机坍塌 + Stage 2 负载重分配级联
- dynamics:   主时间步循环与指标记录
"""
