"""
基本量子门示例
这个文件演示了 Qiskit 中的基本量子门操作
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def pauli_gates_demo():
    """演示泡利门 (Pauli Gates): X, Y, Z"""
    print("=== 泡利门演示 ===")
    
    # X门 (比特翻转)
    qc_x = QuantumCircuit(1, 1)
    qc_x.x(0)  # 应用X门
    qc_x.measure(0, 0)
    print("X门电路 (应该测量到 |1⟩):")
    print(qc_x)
    
    # Y门
    qc_y = QuantumCircuit(1, 1)
    qc_y.y(0)  # 应用Y门
    qc_y.measure(0, 0)
    print("\nY门电路:")
    print(qc_y)
    
    # Z门 (相位翻转)
    qc_z = QuantumCircuit(1, 1)
    qc_z.z(0)  # 应用Z门
    qc_z.measure(0, 0)
    print("\nZ门电路 (应该测量到 |0⟩):")
    print(qc_z)
    
    # 运行所有电路
    simulator = AerSimulator()
    circuits = [qc_x, qc_y, qc_z]
    names = ["X门", "Y门", "Z门"]
    
    for qc, name in zip(circuits, names):
        job = simulator.run(transpile(qc, simulator), shots=1000)
        result = job.result()
        counts = result.get_counts(qc)
        print(f"{name} 结果: {counts}")


def hadamard_demo():
    """演示Hadamard门创建叠加态"""
    print("\n=== Hadamard门演示 ===")
    
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # 创建叠加态
    qc.measure(0, 0)
    
    print("Hadamard门电路 (应该得到50%的0和50%的1):")
    print(qc)
    
    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    print(f"结果: {counts}")


def cnot_demo():
    """演示CNOT门创建纠缠"""
    print("\n=== CNOT门演示 ===")
    
    qc = QuantumCircuit(2, 2)
    qc.h(0)      # 对控制量子比特创建叠加
    qc.cx(0, 1)  # CNOT门：控制比特是0，目标比特是1
    qc.measure([0, 1], [0, 1])
    
    print("CNOT门电路 (创建纠缠态):")
    print(qc)
    
    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=1000)
    result = job.result()
    counts = result.get_counts(qc)
    print(f"纠缠态结果: {counts}")
    print("注意：应该只看到 '00' 和 '11' 状态，这表明量子比特是纠缠的")


def rotation_gates_demo():
    """演示旋转门"""
    print("\n=== 旋转门演示 ===")
    
    import math
    
    # RX门 (绕X轴旋转)
    qc_rx = QuantumCircuit(1, 1)
    qc_rx.rx(math.pi/2, 0)  # 旋转π/2弧度
    qc_rx.measure(0, 0)
    
    # RY门 (绕Y轴旋转)
    qc_ry = QuantumCircuit(1, 1)
    qc_ry.ry(math.pi/2, 0)
    qc_ry.measure(0, 0)
    
    # RZ门 (绕Z轴旋转)
    qc_rz = QuantumCircuit(1, 1)
    qc_rz.rz(math.pi/2, 0)
    qc_rz.measure(0, 0)
    
    simulator = AerSimulator()
    circuits = [qc_rx, qc_ry, qc_rz]
    names = ["RX(π/2)", "RY(π/2)", "RZ(π/2)"]
    
    for qc, name in zip(circuits, names):
        job = simulator.run(transpile(qc, simulator), shots=1000)
        result = job.result()
        counts = result.get_counts(qc)
        print(f"{name} 结果: {counts}")


if __name__ == "__main__":
    print("量子门基础示例")
    print("="*50)
    
    pauli_gates_demo()
    hadamard_demo()
    cnot_demo()
    rotation_gates_demo() 