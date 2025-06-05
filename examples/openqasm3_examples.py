"""
OpenQASM 3.0 与 Qiskit 交互示例
演示如何在 Qiskit 中使用 OpenQASM 3.0 量子汇编语言
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.qasm3 import loads, dumps


def basic_openqasm3_example():
    """基础 OpenQASM 3.0 示例"""
    print("=== 基础 OpenQASM 3.0 示例 ===")
    
    # 定义一个简单的 OpenQASM 3.0 程序
    qasm3_code = """
    OPENQASM 3.0;
    include "stdgates.inc";
    
    qubit[1] q;
    bit[1] c;
    
    h q[0];
    measure q[0] -> c[0];
    """
    
    print("OpenQASM 3.0 代码:")
    print(qasm3_code)
    
    # 从 OpenQASM 3.0 代码创建 Qiskit 电路
    qc = loads(qasm3_code)
    
    print("\n转换为 Qiskit 电路:")
    print(qc)
    
    # 运行电路
    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    
    print(f"\n测量结果: {counts}")
    return qc, counts


def bell_state_openqasm3():
    """使用 OpenQASM 3.0 创建贝尔态"""
    print("\n=== OpenQASM 3.0 贝尔态示例 ===")
    
    qasm3_bell = """
    OPENQASM 3.0;
    include "stdgates.inc";
    
    qubit[2] q;
    bit[2] c;
    
    // 创建贝尔态
    h q[0];
    cx q[0], q[1];
    
    // 测量两个量子比特
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """
    
    print("贝尔态 OpenQASM 3.0 代码:")
    print(qasm3_bell)
    
    # 转换为 Qiskit 电路
    qc = loads(qasm3_bell)
    print("\n转换为 Qiskit 电路:")
    print(qc)
    
    # 运行模拟
    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    
    print(f"\n贝尔态测量结果: {counts}")
    return qc, counts


def qiskit_to_openqasm3():
    """将 Qiskit 电路转换为 OpenQASM 3.0"""
    print("\n=== Qiskit 电路转换为 OpenQASM 3.0 ===")
    
    # 创建一个 Qiskit 电路
    qc = QuantumCircuit(3, 3)
    
    # 添加一些门操作
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.rz(0.5, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    
    print("原始 Qiskit 电路:")
    print(qc)
    
    # 转换为 OpenQASM 3.0
    qasm3_str = dumps(qc)
    print("\n转换为 OpenQASM 3.0:")
    print(qasm3_str)
    
    # 验证可以重新加载
    qc_reloaded = loads(qasm3_str)
    print("\n重新加载的电路:")
    print(qc_reloaded)
    
    return qc, qasm3_str


def advanced_openqasm3_features():
    """展示 OpenQASM 3.0 的高级特性"""
    print("\n=== OpenQASM 3.0 高级特性 ===")
    
    # 使用变量和函数的 OpenQASM 3.0 程序
    qasm3_advanced = """
    OPENQASM 3.0;
    include "stdgates.inc";
    
    // 定义量子比特和经典比特
    qubit[3] q;
    bit[3] c;
    
    // 定义一个角度变量
    angle theta = pi/4;
    
    // 应用旋转门
    h q[0];
    ry(theta) q[1];
    rz(theta * 2) q[2];
    
    // 创建纠缠
    cx q[0], q[1];
    cx q[1], q[2];
    
    // 测量
    measure q -> c;
    """
    
    print("高级 OpenQASM 3.0 代码 (使用变量和函数):")
    print(qasm3_advanced)
    
    # 注意：这个示例可能需要更新版本的 qiskit 来支持所有特性
    try:
        qc = loads(qasm3_advanced)
        print("\n转换为 Qiskit 电路:")
        print(qc)
        
        # 运行模拟
        simulator = AerSimulator()
        job = simulator.run(transpile(qc, simulator), shots=1000)
        result = job.result()
        counts = result.get_counts(qc)
        print(f"\n测量结果: {counts}")
        
    except Exception as e:
        print(f"\n注意：某些高级特性可能需要更新的 Qiskit 版本")
        print(f"错误信息: {e}")
        
        # 创建一个简化版本
        qc_simple = QuantumCircuit(3, 3)
        qc_simple.h(0)
        qc_simple.ry(3.14159/4, 1)
        qc_simple.rz(3.14159/2, 2)
        qc_simple.cx(0, 1)
        qc_simple.cx(1, 2)
        qc_simple.measure([0, 1, 2], [0, 1, 2])
        
        print("\n简化版本的电路:")
        print(qc_simple)
        
        # 运行简化版本
        simulator = AerSimulator()
        job = simulator.run(transpile(qc_simple, simulator), shots=1000)
        result = job.result()
        counts = result.get_counts(qc_simple)
        print(f"\n简化版本测量结果: {counts}")


def openqasm3_file_operations():
    """演示 OpenQASM 3.0 文件操作"""
    print("\n=== OpenQASM 3.0 文件操作 ===")
    
    # 创建一个简单的电路并保存为 OpenQASM 3.0 文件
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    
    # 转换为 OpenQASM 3.0 并保存到文件
    qasm3_content = dumps(qc)
    
    with open("examples/bell_state.qasm", "w") as f:
        f.write(qasm3_content)
    
    print("已将贝尔态电路保存为 'examples/bell_state.qasm'")
    print("文件内容:")
    print(qasm3_content)
    
    # 从文件读取并重新创建电路
    with open("examples/bell_state.qasm", "r") as f:
        loaded_qasm = f.read()
    
    qc_loaded = loads(loaded_qasm)
    print("\n从文件加载的电路:")
    print(qc_loaded)
    
    return qc_loaded


if __name__ == "__main__":
    print("OpenQASM 3.0 与 Qiskit 交互示例")
    print("="*50)
    
    # 运行所有示例
    basic_openqasm3_example()
    bell_state_openqasm3()
    qiskit_to_openqasm3()
    advanced_openqasm3_features()
    openqasm3_file_operations()
    
    print("\n=== 总结 ===")
    print("OpenQASM 3.0 的主要特性:")
    print("1. 更清晰的语法和结构")
    print("2. 支持变量和函数")
    print("3. 更好的类型系统")
    print("4. 与 Qiskit 的无缝集成")
    print("5. 标准化的量子程序表示") 