#!/usr/bin/env python3
"""
验证门分解的正确性
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_learning.ion_trap.compiler import RThetaPhiGate, MSGate, VirtualZGate

def r_matrix(theta, phi):
    """R(θ, φ) 门矩阵"""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -1j * s * np.exp(-1j * phi)],
        [-1j * s * np.exp(1j * phi), c]
    ])

def rz_matrix(phi):
    """RZ门矩阵"""
    return np.array([
        [np.exp(-1j * phi / 2), 0],
        [0, np.exp(1j * phi / 2)]
    ])

def test_hadamard_decomposition():
    """测试Hadamard门分解"""
    print("=== Hadamard门分解验证 ===")
    
    # 标准Hadamard
    H_standard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    print("标准Hadamard矩阵:")
    print(H_standard)
    
    # 我们的分解: H = RZ(π/2) * RY(π/2) * RZ(π/2)
    rz1 = rz_matrix(np.pi/2)
    ry = r_matrix(np.pi/2, np.pi/2)  # RY(π/2)
    rz2 = rz_matrix(np.pi/2)
    
    H_decomposed = rz2 @ ry @ rz1
    print("\n分解后的Hadamard矩阵:")
    print(H_decomposed)
    
    # 检查是否相等（考虑全局相位）
    diff = np.linalg.norm(H_standard - H_decomposed)
    print(f"\n差异（直接比较）: {diff:.10f}")
    
    # 检查相位差异
    for phase in [1, -1, 1j, -1j]:
        diff_phase = np.linalg.norm(H_standard - phase * H_decomposed)
        print(f"差异（相位 {phase}）: {diff_phase:.10f}")

def test_ms_gate():
    """测试MS门"""
    print("\n=== MS门测试 ===")
    
    # 创建MS门
    ms_gate = MSGate(phi=0.0, theta=np.pi/2)
    qc = QuantumCircuit(2)
    qc.append(ms_gate, [0, 1])
    
    ms_matrix = Operator(qc).data
    print("MS门矩阵:")
    print(ms_matrix)
    
    # 验证MS门的性质
    # MS门应该是酉矩阵
    is_unitary = np.allclose(ms_matrix @ ms_matrix.conj().T, np.eye(4))
    print(f"是否为酉矩阵: {is_unitary}")

def test_cnot_decomposition():
    """测试CNOT门分解"""
    print("\n=== CNOT门分解验证 ===")
    
    # 标准CNOT门
    cnot_standard = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ])
    print("标准CNOT矩阵:")
    print(cnot_standard)
    
    # 我们的分解: Ry(-π/2) ⊗ Ry(-π/2) · MS(π/2) · Ry(π/2) ⊗ I
    
    # 单量子比特门
    ry_neg = r_matrix(np.pi/2, np.pi/2)  # Ry(-π/2) - 注意我们这里实际上是R(π/2, π/2)
    ry_pos = r_matrix(np.pi/2, -np.pi/2)  # Ry(π/2) - 实际上是R(π/2, -π/2)
    I = np.eye(2)
    
    # 预处理: Ry(-π/2) ⊗ Ry(-π/2)
    pre_processing = np.kron(ry_neg, ry_neg)
    
    # MS门矩阵
    c = np.cos(np.pi/4)  # theta = π/2, so cos(θ/2) = cos(π/4)
    s = np.sin(np.pi/4)  # sin(θ/2) = sin(π/4)
    ms_matrix = np.array([
        [c, 0, 0, -1j * s],
        [0, c, -1j * s, 0],
        [0, -1j * s, c, 0],
        [-1j * s, 0, 0, c]
    ])
    
    # 后处理: Ry(π/2) ⊗ Ry(π/2)
    post_processing = np.kron(ry_pos, ry_pos)
    
    # 完整分解
    cnot_decomposed = post_processing @ ms_matrix @ pre_processing
    
    print("\n分解后的CNOT矩阵:")
    print(cnot_decomposed)
    
    # 检查差异
    diff = np.linalg.norm(cnot_standard - cnot_decomposed)
    print(f"\n差异（直接比较）: {diff:.10f}")
    
    # 检查相位差异
    for phase in [1, -1, 1j, -1j]:
        diff_phase = np.linalg.norm(cnot_standard - phase * cnot_decomposed)
        print(f"差异（相位 {phase}）: {diff_phase:.10f}")

def test_bell_state():
    """测试Bell态制备"""
    print("\n=== Bell态测试 ===")
    
    # 标准Bell态: |00⟩ + |11⟩
    bell_standard = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    
    # 使用标准门制备
    qc_standard = QuantumCircuit(2)
    qc_standard.h(0)
    qc_standard.cx(0, 1)
    
    from qiskit.quantum_info import Statevector
    bell_qiskit = Statevector.from_instruction(qc_standard).data
    
    print("Qiskit Bell态:")
    print(bell_qiskit)
    print("理论Bell态:")
    print(bell_standard)
    
    diff = np.linalg.norm(bell_qiskit - bell_standard)
    print(f"差异: {diff:.10f}")

if __name__ == "__main__":
    test_hadamard_decomposition()
    test_ms_gate()
    test_cnot_decomposition()
    test_bell_state() 