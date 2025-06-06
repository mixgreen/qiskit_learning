#!/usr/bin/env python3
"""
快速门验证脚本
==============

专门用于验证特定量子门在离子阱编译器中的正确性
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from ion_trap_compiler import IonTrapCompiler

def verify_gate(gate_name: str, original_circuit: QuantumCircuit, tolerance: float = 1e-10) -> bool:
    """
    验证单个门的正确性
    
    Args:
        gate_name: 门的名称
        original_circuit: 原始电路
        tolerance: 容忍度
        
    Returns:
        是否通过验证
    """
    print(f"\n验证 {gate_name}:")
    print("-" * 30)
    
    # 编译电路
    compiler = IonTrapCompiler()
    compiled_circuit = compiler.compile_circuit(original_circuit)
    
    # 获取状态向量
    try:
        original_state = Statevector.from_instruction(original_circuit).data
        compiled_state = Statevector.from_instruction(compiled_circuit).data
    except Exception as e:
        print(f"❌ 电路运行失败: {e}")
        return False
    
    # 计算保真度
    fidelity = abs(np.vdot(original_state, compiled_state))**2
    
    # 检查是否考虑全局相位后相等
    phases = [1, -1, 1j, -1j]
    min_distance = float('inf')
    
    for phase in phases:
        distance = np.linalg.norm(original_state - phase * compiled_state)
        min_distance = min(min_distance, distance)
    
    is_correct = min_distance < tolerance
    
    # 打印结果
    print(f"原始态: {original_state}")
    print(f"编译态: {compiled_state}")
    print(f"保真度: {fidelity:.10f}")
    print(f"最小距离: {min_distance:.2e}")
    print(f"结果: {'✅ 通过' if is_correct else '❌ 失败'}")
    
    # 显示门分解
    print(f"\n门分解信息:")
    print(f"原始门数: {original_circuit.count_ops()}")
    print(f"编译后门数: {compiled_circuit.count_ops()}")
    print(f"编译后电路:")
    print(compiled_circuit.draw())
    
    return is_correct

def test_basic_gates():
    """测试基本门"""
    print("离子阱编译器基本门验证")
    print("=" * 50)
    
    results = {}
    
    # 测试X门
    qc = QuantumCircuit(1)
    qc.x(0)
    results['X门'] = verify_gate('X门', qc)
    
    # 测试Y门
    qc = QuantumCircuit(1)
    qc.y(0)
    results['Y门'] = verify_gate('Y门', qc)
    
    # 测试Z门
    qc = QuantumCircuit(1)
    qc.z(0)
    results['Z门'] = verify_gate('Z门', qc)
    
    # 测试H门
    qc = QuantumCircuit(1)
    qc.h(0)
    results['H门'] = verify_gate('H门', qc)
    
    # 测试S门
    qc = QuantumCircuit(1)
    qc.s(0)
    results['S门'] = verify_gate('S门', qc)
    
    # 测试T门
    qc = QuantumCircuit(1)
    qc.t(0)
    results['T门'] = verify_gate('T门', qc)
    
    # 测试CNOT门
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    results['CNOT门'] = verify_gate('CNOT门', qc)
    
    # 测试CZ门
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    results['CZ门'] = verify_gate('CZ门', qc)
    
    # 总结
    print("\n" + "=" * 50)
    print("验证总结:")
    passed = sum(results.values())
    total = len(results)
    print(f"通过: {passed}/{total} ({passed/total*100:.1f}%)")
    
    for gate, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {gate}")
    
    return results

def test_rotation_gates():
    """测试旋转门"""
    print("\n旋转门验证")
    print("=" * 30)
    
    results = {}
    
    # RX门
    qc = QuantumCircuit(1)
    qc.rx(np.pi/4, 0)
    results['RX(π/4)'] = verify_gate('RX(π/4)', qc)
    
    # RY门
    qc = QuantumCircuit(1)
    qc.ry(np.pi/3, 0)
    results['RY(π/3)'] = verify_gate('RY(π/3)', qc)
    
    # RZ门
    qc = QuantumCircuit(1)
    qc.rz(np.pi/6, 0)
    results['RZ(π/6)'] = verify_gate('RZ(π/6)', qc)
    
    return results

def test_composite_circuits():
    """测试复合电路"""
    print("\n复合电路验证")
    print("=" * 30)
    
    results = {}
    
    # Bell态
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    results['Bell态'] = verify_gate('Bell态', qc)
    
    # 三量子比特GHZ态
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    results['GHZ态'] = verify_gate('GHZ态', qc)
    
    return results

def manual_state_comparison():
    """手动状态比较示例"""
    print("\n手动状态比较示例")
    print("=" * 30)
    
    # 创建一个简单的H门电路
    qc = QuantumCircuit(1)
    qc.h(0)
    
    # 理论上的Hadamard门结果
    expected_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    
    # 编译并运行
    compiler = IonTrapCompiler()
    compiled_qc = compiler.compile(qc)
    actual_state = Statevector.from_instruction(compiled_qc).data
    
    print("Hadamard门状态比较:")
    print(f"期望状态: {expected_state}")
    print(f"实际状态: {actual_state}")
    print(f"差异: {np.linalg.norm(expected_state - actual_state):.2e}")
    
    # 考虑全局相位
    for phase in [1, -1, 1j, -1j]:
        diff = np.linalg.norm(expected_state - phase * actual_state)
        print(f"相位 {phase}: 差异 = {diff:.2e}")

def compare_matrix_representations():
    """比较矩阵表示"""
    print("\n矩阵表示比较")
    print("=" * 30)
    
    # X门矩阵比较
    qc_x = QuantumCircuit(1)
    qc_x.x(0)
    
    compiler = IonTrapCompiler()
    compiled_x = compiler.compile(qc_x)
    
    # 获取unitary矩阵
    original_matrix = Operator(qc_x).data
    compiled_matrix = Operator(compiled_x).data
    
    print("X门矩阵比较:")
    print("原始矩阵:")
    print(original_matrix)
    print("编译后矩阵:")
    print(compiled_matrix)
    print(f"矩阵差异: {np.linalg.norm(original_matrix - compiled_matrix):.2e}")

def main():
    """主函数"""
    print("快速门验证程序")
    print("=" * 50)
    
    # 运行基本门测试
    basic_results = test_basic_gates()
    
    # 运行旋转门测试
    rotation_results = test_rotation_gates()
    
    # 运行复合电路测试
    composite_results = test_composite_circuits()
    
    # 手动比较示例
    manual_state_comparison()
    
    # 矩阵表示比较
    compare_matrix_representations()
    
    # 总体统计
    all_results = {**basic_results, **rotation_results, **composite_results}
    total_passed = sum(all_results.values())
    total_tests = len(all_results)
    
    print(f"\n" + "=" * 50)
    print("最终总结:")
    print(f"总通过率: {total_passed}/{total_tests} ({total_passed/total_tests*100:.1f}%)")
    
    if total_passed == total_tests:
        print("🎉 所有测试通过！编译器工作正常。")
    else:
        print("⚠️  部分测试失败，需要检查编译器实现。")
        failed_tests = [name for name, passed in all_results.items() if not passed]
        print(f"失败的测试: {', '.join(failed_tests)}")

if __name__ == "__main__":
    main() 