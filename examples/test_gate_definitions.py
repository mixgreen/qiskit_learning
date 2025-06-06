#!/usr/bin/env python3
"""
测试标准量子门的矩阵定义
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

def print_gate_matrix(gate_name, matrix):
    """打印门的矩阵"""
    print(f"\n{gate_name} 门矩阵:")
    print(matrix)
    print(f"行列式: {np.linalg.det(matrix):.6f}")

def test_single_qubit_gates():
    """测试单量子比特门"""
    print("=== 单量子比特门矩阵 ===")
    
    # Hadamard门
    qc = QuantumCircuit(1)
    qc.h(0)
    h_matrix = Operator(qc).data
    print_gate_matrix("Hadamard", h_matrix)
    
    # X门
    qc = QuantumCircuit(1)
    qc.x(0)
    x_matrix = Operator(qc).data
    print_gate_matrix("Pauli-X", x_matrix)
    
    # Y门
    qc = QuantumCircuit(1)
    qc.y(0)
    y_matrix = Operator(qc).data
    print_gate_matrix("Pauli-Y", y_matrix)
    
    # Z门
    qc = QuantumCircuit(1)
    qc.z(0)
    z_matrix = Operator(qc).data
    print_gate_matrix("Pauli-Z", z_matrix)
    
    # RX(π/2)
    qc = QuantumCircuit(1)
    qc.rx(np.pi/2, 0)
    rx_matrix = Operator(qc).data
    print_gate_matrix("RX(π/2)", rx_matrix)
    
    # RY(π/2)
    qc = QuantumCircuit(1)
    qc.ry(np.pi/2, 0)
    ry_matrix = Operator(qc).data
    print_gate_matrix("RY(π/2)", ry_matrix)
    
    # RZ(π/2)
    qc = QuantumCircuit(1)
    qc.rz(np.pi/2, 0)
    rz_matrix = Operator(qc).data
    print_gate_matrix("RZ(π/2)", rz_matrix)

def test_r_theta_phi_gate():
    """测试R(theta, phi)门的定义"""
    print("\n=== R(θ, φ) 门测试 ===")
    
    def r_matrix(theta, phi):
        """R(θ, φ) 门矩阵"""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([
            [c, -1j * s * np.exp(-1j * phi)],
            [-1j * s * np.exp(1j * phi), c]
        ])
    
    # 测试不同参数
    test_cases = [
        ("R(π, 0) - 应该等于X", np.pi, 0),
        ("R(π, π/2) - 应该等于Y", np.pi, np.pi/2),
        ("R(π/2, π/2) - RY(π/2)", np.pi/2, np.pi/2),
        ("R(π/2, 0) - RX(π/2)", np.pi/2, 0),
    ]
    
    for name, theta, phi in test_cases:
        matrix = r_matrix(theta, phi)
        print_gate_matrix(name, matrix)

def test_hadamard_decomposition():
    """测试Hadamard门的分解"""
    print("\n=== Hadamard门分解测试 ===")
    
    # 标准Hadamard矩阵
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    print_gate_matrix("标准Hadamard", H)
    
    # 尝试不同的分解
    def r_matrix(theta, phi):
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([
            [c, -1j * s * np.exp(-1j * phi)],
            [-1j * s * np.exp(1j * phi), c]
        ])
    
    # 分解1: H = RY(π/2) * RZ(π)
    ry_pi2 = r_matrix(np.pi/2, np.pi/2)
    rz_pi = np.array([[np.exp(-1j*np.pi/2), 0], [0, np.exp(1j*np.pi/2)]])
    h_decomp1 = ry_pi2 @ rz_pi
    print_gate_matrix("RY(π/2) * RZ(π)", h_decomp1)
    
    # 分解2: H = RZ(π/2) * RY(π/2) * RZ(π/2)
    rz_pi2 = np.array([[np.exp(-1j*np.pi/4), 0], [0, np.exp(1j*np.pi/4)]])
    h_decomp2 = rz_pi2 @ ry_pi2 @ rz_pi2
    print_gate_matrix("RZ(π/2) * RY(π/2) * RZ(π/2)", h_decomp2)
    
    # 检查是否相等（考虑全局相位）
    print(f"\n标准H与分解1的差异: {np.linalg.norm(H - h_decomp1):.6f}")
    print(f"标准H与分解2的差异: {np.linalg.norm(H - h_decomp2):.6f}")

def test_cnot_matrix():
    """测试CNOT门矩阵"""
    print("\n=== CNOT门矩阵 ===")
    
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    cnot_matrix = Operator(qc).data
    print_gate_matrix("CNOT", cnot_matrix)

if __name__ == "__main__":
    test_single_qubit_gates()
    test_r_theta_phi_gate()
    test_hadamard_decomposition()
    test_cnot_matrix() 