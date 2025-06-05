"""
Qiskit 学习项目主文件
这个文件包含一些基本的 qiskit 示例，帮助你开始学习量子计算
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from qiskit.qasm3 import dumps
import matplotlib.pyplot as plt


def hello_quantum():
    """创建一个简单的量子电路，演示量子叠加"""
    print("=== Hello Quantum World! ===")
    
    # 创建一个包含1个量子比特和1个经典比特的量子电路
    qc = QuantumCircuit(1, 1)
    
    # 应用 Hadamard 门创建叠加态
    qc.h(0)
    
    # 测量量子比特
    qc.measure(0, 0)
    
    print("量子电路:")
    print(qc)
    
    # 显示对应的 OpenQASM 3.0 代码
    print("\n对应的 OpenQASM 3.0 代码:")
    print(dumps(qc))
    
    # 使用 Aer 模拟器运行电路
    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    
    print(f"\n测量结果 (1000次运行): {counts}")
    return qc, counts


def bell_state_example():
    """创建贝尔态，演示量子纠缠"""
    print("\n=== 贝尔态示例 ===")
    
    # 创建2量子比特电路
    qc = QuantumCircuit(2, 2)
    
    # 创建贝尔态
    qc.h(0)        # 对第一个量子比特应用Hadamard门
    qc.cx(0, 1)    # 应用CNOT门创建纠缠
    
    # 测量两个量子比特
    qc.measure([0, 1], [0, 1])
    
    print("贝尔态电路:")
    print(qc)
    
    # 显示对应的 OpenQASM 3.0 代码
    print("\n对应的 OpenQASM 3.0 代码:")
    print(dumps(qc))
    
    # 运行模拟
    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    
    print(f"\n贝尔态测量结果: {counts}")
    return qc, counts


def main():
    """主函数，运行所有示例"""
    print("欢迎来到 Qiskit 学习项目!")
    print("这个项目将帮助你学习量子计算的基础概念\n")
    
    # 运行示例
    qc1, counts1 = hello_quantum()
    qc2, counts2 = bell_state_example()
    
    print("\n=== 项目设置完成 ===")
    print("你可以:")
    print("1. 运行 'uv run jupyter notebook' 启动 Jupyter Notebook")
    print("2. 查看 examples/ 目录中的更多示例")
    print("3. 运行 'uv run python examples/openqasm3_examples.py' 学习 OpenQASM 3.0")
    print("4. 运行 'uv run python examples/ion_trap_compiler.py' 学习离子阱编译器")
    print("5. 运行 'uv run python examples/ion_trap_usage_example.py' 查看离子阱使用示例")
    print("6. 运行 'uv run python examples/ion_trap_measurement_demo.py' 学习离子阱测量特性")
    print("7. 修改这个文件来尝试自己的量子电路")


if __name__ == "__main__":
    main()
