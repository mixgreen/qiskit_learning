"""
离子阱测量特性演示
展示离子阱系统中测量操作的特殊性和处理方式
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# 导入离子阱编译器
from qiskit_learning.ion_trap.compiler import IonTrapCompiler


def demo_final_measurement():
    """演示最终测量（推荐方式）"""
    print("=== 离子阱最终测量演示 ===")
    print("这是离子阱系统推荐的测量方式\n")
    
    # 创建电路：所有量子操作在前，测量在最后
    qc = QuantumCircuit(3, 3, name="final_measurement")
    
    # 量子操作
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rz(np.pi/4, 0)
    
    # 最终测量（推荐：所有测量在电路末尾）
    qc.measure([0, 1, 2], [0, 1, 2])
    
    print("原始电路（推荐方式）:")
    print(qc)
    print()
    
    # 编译
    compiler = IonTrapCompiler(optimization_level=1)
    ion_qc = compiler.compile_circuit(qc)
    
    print("编译后的离子阱电路:")
    print(ion_qc)
    print()
    
    return qc, ion_qc


def demo_mid_circuit_measurement():
    """演示中间测量（不推荐，会有警告）"""
    print("=== 离子阱中间测量演示 ===")
    print("展示离子阱系统如何处理不支持的中间测量\n")
    
    # 创建包含中间测量的电路
    qc = QuantumCircuit(3, 3, name="mid_circuit_measurement")
    
    # 第一阶段操作
    qc.h(0)
    qc.cx(0, 1)
    
    # 中间测量（不推荐）
    qc.measure(0, 0)
    
    # 基于测量结果的后续操作（在离子阱中会有问题）
    qc.h(1)
    qc.cx(1, 2)
    
    # 最终测量
    qc.measure([1, 2], [1, 2])
    
    print("包含中间测量的电路（不推荐）:")
    print(qc)
    print()
    
    # 编译（会产生警告）
    print("编译过程中的警告信息:")
    compiler = IonTrapCompiler(optimization_level=1)
    ion_qc = compiler.compile_circuit(qc)
    
    print("\n编译后的离子阱电路:")
    print(ion_qc)
    print()
    
    return qc, ion_qc


def demo_global_measurement():
    """演示全局测量特性"""
    print("=== 离子阱全局测量特性演示 ===")
    print("离子阱可以同时测量多个离子\n")
    
    # 创建多量子比特纠缠电路
    qc = QuantumCircuit(4, 4, name="global_measurement")
    
    # 创建4量子比特GHZ态
    qc.h(0)
    for i in range(3):
        qc.cx(i, i+1)
    
    # 全局测量所有量子比特
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])
    
    print("4量子比特GHZ态电路:")
    print(qc)
    print()
    
    # 编译
    compiler = IonTrapCompiler(optimization_level=1)
    ion_qc = compiler.compile_circuit(qc)
    
    print("编译后显示全局测量信息:")
    print(ion_qc)
    print()
    
    return qc, ion_qc


def demo_measurement_optimization():
    """演示测量相关的优化"""
    print("=== 离子阱测量优化演示 ===")
    print("展示如何优化包含测量的电路\n")
    
    # 创建需要优化的电路
    qc = QuantumCircuit(3, 3, name="measurement_optimization")
    
    # 一些操作
    qc.h(0)
    qc.rz(np.pi/8, 0)  # 这会变成Virtual Z
    qc.cx(0, 1)
    qc.rz(np.pi/4, 1)  # 这也会变成Virtual Z
    qc.h(2)
    qc.rz(np.pi/6, 2)  # 这也会变成Virtual Z
    
    # 最终测量
    qc.measure([0, 1, 2], [0, 1, 2])
    
    print("需要优化的电路:")
    print(qc)
    print()
    
    # 不同优化级别的比较
    for opt_level in [0, 1, 2]:
        print(f"优化级别 {opt_level}:")
        compiler = IonTrapCompiler(optimization_level=opt_level)
        ion_qc = compiler.compile_circuit(qc)
        
        stats = compiler.get_gate_statistics(ion_qc)
        print(f"门统计: {stats}")
        print()
    
    return qc


def demo_future_mid_circuit_support():
    """演示未来中间测量支持的预留接口"""
    print("=== 未来中间测量支持演示 ===")
    print("展示预留的中间测量接口（当前禁用）\n")
    
    # 创建包含中间测量的电路
    qc = QuantumCircuit(2, 2, name="future_mid_circuit")
    qc.h(0)
    qc.measure(0, 0)  # 中间测量
    qc.cx(0, 1)  # 基于测量结果的操作
    qc.measure(1, 1)
    
    print("包含中间测量的电路:")
    print(qc)
    print()
    
    # 尝试启用中间测量支持（当前会警告）
    print("尝试启用中间测量支持:")
    compiler = IonTrapCompiler(
        optimization_level=1, 
        allow_mid_circuit_measurement=True  # 启用预留接口
    )
    
    ion_qc = compiler.compile_circuit(qc)
    
    print("编译结果:")
    print(ion_qc)
    print()
    
    return qc, ion_qc


def compare_measurement_approaches():
    """比较不同测量方法的性能"""
    print("=== 测量方法性能比较 ===")
    print("比较不同测量策略的编译结果\n")
    
    # 方法1：推荐的最终测量
    qc1 = QuantumCircuit(3, 3, name="final_only")
    qc1.h([0, 1, 2])
    qc1.cx(0, 1)
    qc1.cx(1, 2)
    qc1.measure([0, 1, 2], [0, 1, 2])
    
    # 方法2：中间测量（不推荐）
    qc2 = QuantumCircuit(3, 3, name="with_mid")
    qc2.h(0)
    qc2.measure(0, 0)  # 中间测量
    qc2.h(1)
    qc2.cx(1, 2)
    qc2.measure([1, 2], [1, 2])
    
    compiler = IonTrapCompiler(optimization_level=1)
    
    print("性能比较:")
    print("-" * 50)
    
    for qc in [qc1, qc2]:
        print(f"\n电路: {qc.name}")
        print(f"原始门数: {len(qc.data)}")
        
        ion_qc = compiler.compile_circuit(qc)
        ion_gates = len([inst for inst in ion_qc.data if inst.operation.name != 'measure'])
        
        print(f"离子阱门数: {ion_gates}")
        print(f"门统计: {compiler.get_gate_statistics(ion_qc)}")


def main():
    """主函数，运行所有演示"""
    print("离子阱测量特性完整演示")
    print("=" * 50)
    print()
    
    # 运行各种演示
    demo_final_measurement()
    print()
    
    demo_mid_circuit_measurement()
    print()
    
    demo_global_measurement()
    print()
    
    demo_measurement_optimization()
    print()
    
    demo_future_mid_circuit_support()
    print()
    
    compare_measurement_approaches()
    
    print("\n" + "=" * 50)
    print("离子阱测量特性总结:")
    print("1. ✅ 支持最终全局测量（推荐）")
    print("2. ⚠️  中间测量会转换为最终测量（有警告）")
    print("3. 🔄 全局测量可同时读取多个离子")
    print("4. 🚧 预留中间测量接口（未来支持）")
    print("5. ⚡ 优化Virtual Z门以减少测量前的操作")
    
    print("\n推荐做法:")
    print("- 将所有量子门操作放在电路前面")
    print("- 将所有测量操作放在电路末尾")
    print("- 利用离子阱的全局测量能力")
    print("- 避免依赖中间测量结果的条件操作")


if __name__ == "__main__":
    main() 