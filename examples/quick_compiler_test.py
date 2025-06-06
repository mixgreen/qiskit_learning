#!/usr/bin/env python3
"""
离子阱编译器快速测试
===================

快速验证离子阱编译器的基本功能是否正确
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_learning.ion_trap.compiler import IonTrapCompiler

def compare_states(state1, state2, tolerance=1e-10):
    """比较两个量子态，考虑全局相位"""
    fidelity = abs(np.vdot(state1, state2))**2
    
    # 尝试不同全局相位
    phases = [1, -1, 1j, -1j]
    min_distance = float('inf')
    
    for phase in phases:
        diff = np.linalg.norm(state1 - phase * state2)
        min_distance = min(min_distance, diff)
    
    return min_distance < tolerance, fidelity

def quick_test():
    """快速测试基本功能"""
    compiler = IonTrapCompiler()
    
    print("离子阱编译器快速测试")
    print("=" * 40)
    
    def create_h_circuit():
        qc = QuantumCircuit(1)
        qc.h(0)
        return qc
    
    def create_x_circuit():
        qc = QuantumCircuit(1)
        qc.x(0)
        return qc
    
    def create_y_circuit():
        qc = QuantumCircuit(1)
        qc.y(0)
        return qc
    
    def create_z_circuit():
        qc = QuantumCircuit(1)
        qc.z(0)
        return qc
    
    def create_rx_circuit():
        qc = QuantumCircuit(1)
        qc.rx(np.pi/2, 0)
        return qc
    
    def create_ry_circuit():
        qc = QuantumCircuit(1)
        qc.ry(np.pi/3, 0)
        return qc
    
    def create_rz_circuit():
        qc = QuantumCircuit(1)
        qc.rz(np.pi/4, 0)
        return qc
    
    def create_cnot_circuit():
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        return qc
    
    def create_cz_circuit():
        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        return qc
    
    def create_bell_circuit():
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        return qc

    test_cases = [
        # 单量子比特门
        ("Hadamard门", create_h_circuit),
        ("X门", create_x_circuit),
        ("Y门", create_y_circuit),
        ("Z门", create_z_circuit),
        
        # 旋转门
        ("RX(π/2)门", create_rx_circuit),
        ("RY(π/3)门", create_ry_circuit),
        ("RZ(π/4)门", create_rz_circuit),
        
        # 双量子比特门
        ("CNOT门", create_cnot_circuit),
        ("CZ门", create_cz_circuit),
        
        # 复合电路
        ("Bell态", create_bell_circuit),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test_name, circuit_func in test_cases:
        try:
            # 创建原始电路
            qc = circuit_func()
            
            # 编译电路
            compiled_qc = compiler.compile_circuit(qc)
            
            # 获取状态向量
            original_state = Statevector.from_instruction(qc).data
            compiled_state = Statevector.from_instruction(compiled_qc).data
            
            # 比较状态
            is_equal, fidelity = compare_states(original_state, compiled_state)
            
            if is_equal:
                print(f"✓ {test_name:<15} 通过 (保真度: {fidelity:.8f})")
                passed += 1
            else:
                print(f"✗ {test_name:<15} 失败 (保真度: {fidelity:.8f})")
                
        except Exception as e:
            print(f"✗ {test_name:<15} 错误: {str(e)}")
    
    print("-" * 40)
    print(f"测试结果: {passed}/{total} 通过 ({passed/total:.1%})")
    
    if passed == total:
        print("🎉 所有基本测试都通过了！编译器工作正常。")
    else:
        print("⚠️  有测试失败，请检查编译器实现。")
    
    return passed == total

def detailed_example():
    """详细示例：展示一个Bell态的编译过程"""
    print("\n" + "=" * 50)
    print("详细示例：Bell态编译")
    print("=" * 50)
    
    # 创建Bell态电路
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    
    print("原始电路:")
    print(qc.draw())
    
    # 编译
    compiler = IonTrapCompiler()
    compiled_qc = compiler.compile_circuit(qc)
    
    print("\n编译后的电路:")
    print(compiled_qc.draw())
    
    # 获取状态向量
    original_state = Statevector.from_instruction(qc).data
    compiled_state = Statevector.from_instruction(compiled_qc).data
    
    print(f"\n原始状态向量:")
    print(f"  {original_state}")
    print(f"\n编译后状态向量:")
    print(f"  {compiled_state}")
    
    # 比较
    is_equal, fidelity = compare_states(original_state, compiled_state)
    print(f"\n状态比较:")
    print(f"  保真度: {fidelity:.10f}")
    print(f"  状态相等: {'是' if is_equal else '否'}")
    
    # 分析编译统计
    original_stats = qc.count_ops()
    compiled_stats = compiled_qc.count_ops()
    
    print(f"\n编译统计:")
    print(f"  原始电路: 深度={qc.depth()}, 门数={original_stats}")
    print(f"  编译电路: 深度={compiled_qc.depth()}, 门数={compiled_stats}")

if __name__ == "__main__":
    success = quick_test()
    detailed_example()
    
    if success:
        print("\n建议运行完整测试套件:")
        print("  python examples/ion_trap_compiler_test.py") 