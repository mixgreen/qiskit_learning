"""
离子阱编译器使用示例
演示如何使用离子阱编译器转换标准量子电路
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# 导入离子阱编译器
from ion_trap_compiler import IonTrapCompiler, MSGate, RThetaPhiGate, VirtualZGate


def simple_usage_example():
    """简单使用示例"""
    print("=== 离子阱编译器简单使用示例 ===\n")
    
    # 1. 创建一个标准的量子电路
    print("步骤1: 创建标准量子电路")
    qc = QuantumCircuit(2, 2)
    qc.h(0)           # Hadamard门
    qc.cx(0, 1)       # CNOT门
    qc.measure([0, 1], [0, 1])
    
    print("原始电路:")
    print(qc)
    print()
    
    # 2. 创建离子阱编译器
    print("步骤2: 创建离子阱编译器")
    compiler = IonTrapCompiler(optimization_level=1)
    print("编译器已创建")
    print()
    
    # 3. 编译电路
    print("步骤3: 编译为离子阱电路")
    ion_circuit = compiler.compile_circuit(qc)
    
    print("编译后的离子阱电路:")
    print(ion_circuit)
    print()
    
    # 4. 分析结果
    print("步骤4: 分析编译结果")
    original_stats = compiler.get_gate_statistics(qc)
    ion_stats = compiler.get_gate_statistics(ion_circuit)
    
    print(f"原始电路门统计: {original_stats}")
    print(f"离子阱电路门统计: {ion_stats}")
    print()
    
    return qc, ion_circuit


def advanced_usage_example():
    """高级使用示例"""
    print("=== 高级使用示例：量子算法编译 ===\n")
    
    # 创建一个复杂的量子算法电路
    print("创建复杂的量子算法电路（模拟 Grover 算法的一部分）")
    qc = QuantumCircuit(3, 3)
    
    # 初始化叠加态
    qc.h([0, 1, 2])
    
    # Oracle (标记目标状态 |110⟩)
    qc.cz(0, 1)
    qc.cz(1, 2)
    
    # Diffuser
    qc.h([0, 1, 2])
    qc.x([0, 1, 2])
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x([0, 1, 2])
    qc.h([0, 1, 2])
    
    qc.measure([0, 1, 2], [0, 1, 2])
    
    print("原始复杂电路:")
    print(qc)
    print()
    
    # 编译为离子阱电路
    compiler = IonTrapCompiler(optimization_level=2)
    ion_circuit = compiler.compile_circuit(qc)
    
    print("编译后的离子阱电路:")
    print(ion_circuit)
    print()
    
    # 统计分析
    original_stats = compiler.get_gate_statistics(qc)
    ion_stats = compiler.get_gate_statistics(ion_circuit)
    
    print("门使用统计比较:")
    print(f"原始电路: {original_stats}")
    print(f"离子阱电路: {ion_stats}")
    
    # 计算门数量的变化
    original_count = sum(original_stats.values())
    ion_count = sum(ion_stats.values())
    print(f"\n总门数变化: {original_count} -> {ion_count}")
    print(f"变化比例: {ion_count/original_count:.2f}x")
    
    return qc, ion_circuit


def custom_gate_usage():
    """自定义门使用示例"""
    print("\n=== 自定义离子阱门使用示例 ===\n")
    
    # 直接使用离子阱门创建电路
    print("直接使用离子阱门创建电路:")
    
    qc = QuantumCircuit(2, 2)
    
    # 添加自定义离子阱门
    r_gate = RThetaPhiGate(np.pi/2, np.pi/4)
    ms_gate = MSGate(phi=0.0, theta=np.pi/2)
    vz_gate = VirtualZGate(np.pi/6)
    
    qc.append(r_gate, [0])
    qc.append(r_gate, [1]) 
    qc.append(ms_gate, [0, 1])
    qc.append(vz_gate, [0])
    qc.append(vz_gate, [1])
    qc.measure([0, 1], [0, 1])
    
    print("使用自定义离子阱门的电路:")
    print(qc)
    print()
    
    # 分析门的参数
    print("门参数分析:")
    for i, instruction in enumerate(qc.data):
        gate = instruction.operation
        if hasattr(gate, 'params') and gate.params:
            print(f"第{i+1}个门 ({gate.name}): 参数 = {gate.params}")
    
    return qc


def performance_comparison():
    """性能比较示例"""
    print("\n=== 性能比较示例 ===\n")
    
    # 创建多个不同复杂度的电路进行比较
    circuits = []
    
    # 简单电路
    qc1 = QuantumCircuit(2, 2, name="simple")
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.measure([0, 1], [0, 1])
    circuits.append(qc1)
    
    # 中等复杂度电路
    qc2 = QuantumCircuit(3, 3, name="medium")
    qc2.h([0, 1, 2])
    qc2.cx(0, 1)
    qc2.cx(1, 2)
    qc2.rz(np.pi/4, 0)
    qc2.ry(np.pi/3, 1)
    qc2.measure([0, 1, 2], [0, 1, 2])
    circuits.append(qc2)
    
    # 复杂电路
    qc3 = QuantumCircuit(4, 4, name="complex")
    for i in range(4):
        qc3.h(i)
    for i in range(3):
        qc3.cx(i, i+1)
    qc3.cz(0, 2)
    qc3.cz(1, 3)
    for i in range(4):
        qc3.rz(np.pi/(i+2), i)
    qc3.measure(range(4), range(4))
    circuits.append(qc3)
    
    # 编译并比较
    compiler = IonTrapCompiler(optimization_level=1)
    
    print("性能比较结果:")
    print("-" * 60)
    print(f"{'电路名称':<10} {'原始门数':<8} {'离子阱门数':<10} {'比例':<8}")
    print("-" * 60)
    
    for qc in circuits:
        ion_qc = compiler.compile_circuit(qc)
        
        orig_count = sum(compiler.get_gate_statistics(qc).values())
        ion_count = sum(compiler.get_gate_statistics(ion_qc).values())
        ratio = ion_count / orig_count
        
        print(f"{qc.name:<10} {orig_count:<8} {ion_count:<10} {ratio:<8.2f}")
    
    print("-" * 60)


def main():
    """主函数，运行所有示例"""
    print("离子阱编译器使用示例集合")
    print("=" * 50)
    
    # 运行各种示例
    simple_usage_example()
    advanced_usage_example()
    custom_gate_usage()
    performance_comparison()
    
    print("\n" + "=" * 50)
    print("使用总结:")
    print("1. 创建 IonTrapCompiler 实例")
    print("2. 使用 compile_circuit() 方法编译标准电路")
    print("3. 分析编译结果和性能")
    print("4. 可以直接使用离子阱门构建电路")
    print("5. 支持不同的优化级别")
    print("6. 理解离子阱测量特性（全局测量 vs 中间测量）")
    
    print("\n离子阱测量特性:")
    print("  ✅ 支持: 最终全局测量（推荐）")
    print("  ⚠️  限制: 中间测量会转换为最终测量")
    print("  🔄 优势: 可同时测量多个离子")
    print("  🚧 预留: 中间测量接口（未来支持）")
    
    print("\n支持的标准门转换:")
    gate_mappings = {
        "H": "R(π, π/2)",
        "X": "R(π, 0)",
        "Y": "R(π, π/2)",
        "Z": "Virtual Z(π)",
        "RX": "R(θ, 0)",
        "RY": "R(θ, π/2)", 
        "RZ": "Virtual Z(φ)",
        "CNOT": "MS门序列",
        "CZ": "MS门序列",
        "S": "Virtual Z(π/2)",
        "T": "Virtual Z(π/4)",
        "Measure": "全局测量（仅最终）"
    }
    
    for original, ion_trap in gate_mappings.items():
        print(f"  {original:<8} -> {ion_trap}")
        
    print(f"\n💡 提示: 运行 'uv run python examples/ion_trap_measurement_demo.py' 查看详细的测量特性演示")


if __name__ == "__main__":
    main() 