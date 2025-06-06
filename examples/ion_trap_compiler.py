"""
离子阱量子计算编译器
将通用量子线路转换为离子阱系统专用的门操作

离子阱系统特有的门：
1. MS门 (Mølmer-Sørensen gate) - 两量子比特纠缠门
2. R_theta_phi门 - 单量子比特旋转门 
3. Virtual_Z门 - 虚拟Z旋转门
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import *
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
import math


class IonTrapGate(Gate):
    """离子阱门的基类"""
    pass


class MSGate(IonTrapGate):
    """Mølmer-Sørensen 门 - 离子阱中的两量子比特纠缠门
    
    MS门是离子阱量子计算中实现两量子比特纠缠的标准门，
    通过激光与离子的相互作用实现。
    """
    
    def __init__(self, phi: float = 0.0, theta: float = np.pi/2):
        """
        初始化MS门
        
        Args:
            phi: 相位参数
            theta: 旋转角度，默认为π/2实现最大纠缠
        """
        self.phi = phi
        self.theta = theta
        super().__init__("ms", 2, [phi, theta])
    
    def _define(self):
        """定义MS门的矩阵表示"""
        # MS门的酉矩阵表示
        c = np.cos(self.theta / 2)
        s = np.sin(self.theta / 2)
        
        # MS门矩阵
        ms_matrix = np.array([
            [c, 0, 0, -1j * s * np.exp(-1j * self.phi)],
            [0, c, -1j * s * np.exp(-1j * self.phi), 0],
            [0, -1j * s * np.exp(1j * self.phi), c, 0],
            [-1j * s * np.exp(1j * self.phi), 0, 0, c]
        ])
        
        self.definition = QuantumCircuit(2)
        self.definition.unitary(ms_matrix, [0, 1], label=f"MS({self.phi:.3f},{self.theta:.3f})")


class RThetaPhiGate(IonTrapGate):
    """R_theta_phi 门 - 离子阱中的单量子比特旋转门
    
    通过激光脉冲实现任意单量子比特旋转
    """
    
    def __init__(self, theta: float, phi: float):
        """
        初始化 R_theta_phi 门
        
        Args:
            theta: 布洛赫球上的极角
            phi: 布洛赫球上的方位角
        """
        self.theta = theta
        self.phi = phi
        super().__init__("r", 1, [theta, phi])
    
    def _define(self):
        """定义 R_theta_phi 门的矩阵表示"""
        c = np.cos(self.theta / 2)
        s = np.sin(self.theta / 2)
        
        # R_theta_phi 门矩阵
        r_matrix = np.array([
            [c, -1j * s * np.exp(-1j * self.phi)],
            [-1j * s * np.exp(1j * self.phi), c]
        ])
        
        self.definition = QuantumCircuit(1)
        self.definition.unitary(r_matrix, [0], label=f"R({self.theta:.3f},{self.phi:.3f})")


class VirtualZGate(IonTrapGate):
    """Virtual Z 门 - 虚拟Z旋转门
    
    通过软件相位调整实现，不需要实际的激光操作，
    可以与其他门合并以优化电路
    """
    
    def __init__(self, phi: float):
        """
        初始化 Virtual Z 门
        
        Args:
            phi: Z旋转的相位角度
        """
        self.phi = phi
        super().__init__("vz", 1, [phi])
    
    def _define(self):
        """定义 Virtual Z 门"""
        self.definition = QuantumCircuit(1)
        self.definition.rz(self.phi, 0)


class IonTrapCompiler:
    """离子阱量子计算编译器
    
    将标准的量子门转换为离子阱系统专用的门操作
    
    离子阱测量特性：
    - 支持全局测量（电路结束后测量所有离子）
    - 暂不支持中间测量（电路中的测量操作）
    """
    
    def __init__(self, optimization_level: int = 1, allow_mid_circuit_measurement: bool = False):
        """
        初始化编译器
        
        Args:
            optimization_level: 优化级别 (0-3)
            allow_mid_circuit_measurement: 是否允许电路中间测量（预留接口，暂时禁用）
        """
        self.optimization_level = optimization_level
        self.allow_mid_circuit_measurement = allow_mid_circuit_measurement
        self.virtual_z_phases = {}  # 跟踪每个量子比特的虚拟Z相位
        self.measurement_operations = []  # 存储测量操作
    
    def compile_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        将标准量子电路编译为离子阱兼容电路
        
        Args:
            circuit: 输入的标准量子电路
            
        Returns:
            编译后的离子阱电路
        """
        # 创建新的离子阱电路
        ion_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        ion_circuit.name = f"ion_trap_{circuit.name}"
        
        # 初始化状态
        self.virtual_z_phases = {i: 0.0 for i in range(circuit.num_qubits)}
        self.measurement_operations = []
        
        # 分析电路中的测量操作
        self._analyze_measurements(circuit)
        
        # 逐个处理原电路中的门（跳过测量门）
        for instruction in circuit.data:
            gate = instruction.operation
            qubits = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
            
            if gate.name == 'measure':
                # 收集测量操作，稍后处理
                clbits = [circuit.find_bit(clbit).index for clbit in instruction.clbits]
                self._handle_measurement(ion_circuit, qubits, clbits, instruction)
            else:
                self._convert_gate(ion_circuit, gate, qubits)
        
        # 应用剩余的虚拟Z门
        self._apply_remaining_virtual_z(ion_circuit)
        
        # 添加离子阱兼容的测量操作
        self._add_ion_trap_measurements(ion_circuit)
        
        # 优化电路
        if self.optimization_level > 0:
            ion_circuit = self._optimize_circuit(ion_circuit)
        
        return ion_circuit
    
    def _convert_gate(self, circuit: QuantumCircuit, gate, qubits: List[int]):
        """将单个门转换为离子阱门"""
        
        if gate.name == 'h':
            # Hadamard门分解: H = RZ(π/2) * RY(π/2) * RZ(π/2)
            # 使用virtual Z门优化
            self._add_virtual_z(qubits[0], np.pi/2)
            self._add_r_gate(circuit, np.pi/2, np.pi/2, qubits[0])  # RY(π/2)
            self._add_virtual_z(qubits[0], np.pi/2)
            
        elif gate.name == 'x':
            # X门 -> R_theta_phi门 (绕X轴旋转π)
            self._add_r_gate(circuit, np.pi, 0, qubits[0])
            
        elif gate.name == 'y':
            # Y门 -> R_theta_phi门 (绕Y轴旋转π)
            self._add_r_gate(circuit, np.pi, np.pi/2, qubits[0])
            
        elif gate.name == 'z':
            # Z门 -> Virtual Z门
            self._add_virtual_z(qubits[0], np.pi)
            
        elif gate.name == 'rx':
            # RX门 -> R_theta_phi门
            theta = gate.params[0]
            self._add_r_gate(circuit, theta, 0, qubits[0])
            
        elif gate.name == 'ry':
            # RY门 -> R_theta_phi门
            theta = gate.params[0]
            self._add_r_gate(circuit, theta, np.pi/2, qubits[0])
            
        elif gate.name == 'rz':
            # RZ门 -> Virtual Z门
            phi = gate.params[0]
            self._add_virtual_z(qubits[0], phi)
            
        elif gate.name == 'cx' or gate.name == 'cnot':
            # CNOT门 -> MS门序列
            self._convert_cnot_to_ms(circuit, qubits[0], qubits[1])
            
        elif gate.name == 'cz':
            # CZ门 -> MS门序列
            self._convert_cz_to_ms(circuit, qubits[0], qubits[1])
            
        elif gate.name == 'phase' or gate.name == 's':
            # S门 -> Virtual Z门
            self._add_virtual_z(qubits[0], np.pi/2)
            
        elif gate.name == 't':
            # T门 -> Virtual Z门
            self._add_virtual_z(qubits[0], np.pi/4)
            
        else:
            print(f"警告: 门 '{gate.name}' 未被支持，将被跳过")
    
    def _analyze_measurements(self, circuit: QuantumCircuit):
        """分析电路中的测量操作"""
        gate_count = 0
        measurement_positions = []
        
        for i, instruction in enumerate(circuit.data):
            if instruction.operation.name == 'measure':
                measurement_positions.append(i)
            else:
                gate_count += 1
        
        # 检查是否所有测量都在电路末尾
        if measurement_positions:
            last_non_measure = -1
            for i, instruction in enumerate(circuit.data):
                if instruction.operation.name != 'measure':
                    last_non_measure = i
            
            # 如果测量不在末尾，发出警告
            for pos in measurement_positions:
                if pos <= last_non_measure:
                    print(f"警告: 离子阱系统不支持电路中间测量 (位置 {pos})，将转换为最终全局测量")
    
    def _handle_measurement(self, ion_circuit: QuantumCircuit, qubits: List[int], 
                          clbits: List[int], instruction):
        """处理测量操作"""
        # 存储测量信息，稍后统一处理
        measurement_info = {
            'qubits': qubits,
            'clbits': clbits,
            'instruction': instruction
        }
        self.measurement_operations.append(measurement_info)
        
        # 预留中间测量接口（目前禁用）
        if self.allow_mid_circuit_measurement:
            # TODO: 未来支持中间测量时的实现
            # self._add_mid_circuit_measurement(ion_circuit, qubits, clbits)
            pass
    
    def _add_ion_trap_measurements(self, ion_circuit: QuantumCircuit):
        """添加离子阱兼容的测量操作
        
        离子阱系统特点：
        1. 支持全局测量（同时测量所有离子）
        2. 通常在电路执行完毕后进行
        3. 测量是破坏性的，一次性读取所有状态
        """
        if not self.measurement_operations:
            return
        
        # 收集所有需要测量的量子比特
        measured_qubits = set()
        qubit_to_clbit = {}
        
        for measurement in self.measurement_operations:
            for q, c in zip(measurement['qubits'], measurement['clbits']):
                measured_qubits.add(q)
                qubit_to_clbit[q] = c
        
        # 离子阱全局测量：同时测量所有相关离子
        if len(measured_qubits) > 1:
            print(f"离子阱全局测量: 同时测量离子 {sorted(measured_qubits)}")
        
        # 添加测量操作（保持原有的单独测量以兼容 Qiskit）
        for qubit in sorted(measured_qubits):
            if qubit in qubit_to_clbit:
                ion_circuit.measure(qubit, qubit_to_clbit[qubit])
    
    # 预留中间测量接口（暂时禁用）
    def _add_mid_circuit_measurement(self, ion_circuit: QuantumCircuit, 
                                   qubits: List[int], clbits: List[int]):
        """
        预留接口：中间测量支持
        
        注意：当前离子阱系统不支持中间测量，此方法仅为未来扩展预留
        
        Args:
            ion_circuit: 离子阱电路
            qubits: 要测量的量子比特
            clbits: 对应的经典比特
        """
        # TODO: 实现中间测量逻辑
        # 可能的实现方式：
        # 1. 使用辅助离子进行状态转移
        # 2. 实现非破坏性测量协议
        # 3. 分割电路为多个阶段
        print("警告: 中间测量功能尚未实现")
        pass
    
    def _add_r_gate(self, circuit: QuantumCircuit, theta: float, phi: float, qubit: int):
        """添加R_theta_phi门，考虑虚拟Z相位"""
        # 应用累积的虚拟Z相位
        if self.virtual_z_phases[qubit] != 0:
            phi += self.virtual_z_phases[qubit]
            self.virtual_z_phases[qubit] = 0
        
        r_gate = RThetaPhiGate(theta, phi)
        circuit.append(r_gate, [qubit])
    
    def _add_virtual_z(self, qubit: int, phi: float):
        """累积虚拟Z相位"""
        self.virtual_z_phases[qubit] += phi
        # 将相位规范化到 [0, 2π)
        self.virtual_z_phases[qubit] %= (2 * np.pi)
    
    def _convert_cnot_to_ms(self, circuit: QuantumCircuit, control: int, target: int):
        """将CNOT门转换为MS门序列
        
        标准分解: CNOT = Ry(-π/2) ⊗ Ry(-π/2) · MS(π/2) · Ry(π/2) ⊗ I
        其中MS门产生最大纠缠态
        """
        
        # 步骤1: 预处理旋转 - 将两个量子比特都旋转到赤道
        self._add_r_gate(circuit, np.pi/2, np.pi/2, control)   # Ry(-π/2) on control 
        self._add_r_gate(circuit, np.pi/2, np.pi/2, target)    # Ry(-π/2) on target
        
        # 步骤2: MS纠缠门 - 产生最大纠缠
        ms_gate = MSGate(phi=0.0, theta=np.pi/2)
        circuit.append(ms_gate, [control, target])
        
        # 步骤3: 后处理旋转 - 只对控制量子比特进行反向旋转
        self._add_r_gate(circuit, np.pi/2, -np.pi/2, control)  # Ry(π/2) on control
        self._add_r_gate(circuit, np.pi/2, -np.pi/2, target)   # Ry(π/2) on target
    
    def _convert_cz_to_ms(self, circuit: QuantumCircuit, control: int, target: int):
        """将CZ门转换为MS门序列"""
        # CZ = H ⊗ I · CNOT · H ⊗ I
        
        # Hadamard on target
        self._add_r_gate(circuit, np.pi, np.pi/2, target)
        
        # CNOT
        self._convert_cnot_to_ms(circuit, control, target)
        
        # Hadamard on target
        self._add_r_gate(circuit, np.pi, np.pi/2, target)
    
    def _apply_remaining_virtual_z(self, circuit: QuantumCircuit):
        """应用剩余的虚拟Z门"""
        for qubit, phase in self.virtual_z_phases.items():
            if abs(phase) > 1e-10:  # 只应用非零相位
                vz_gate = VirtualZGate(phase)
                circuit.append(vz_gate, [qubit])
        
        # 清空相位记录
        self.virtual_z_phases = {qubit: 0.0 for qubit in self.virtual_z_phases}
    
    def _optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """优化离子阱电路"""
        if self.optimization_level == 0:
            return circuit
        
        # 基本优化：合并相邻的虚拟Z门
        optimized = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        optimized.name = circuit.name
        
        # 这里可以添加更复杂的优化算法
        # 目前只是简单复制
        for instruction in circuit.data:
            optimized.append(instruction.operation, instruction.qubits, instruction.clbits)
        
        return optimized
    
    def get_gate_statistics(self, circuit: QuantumCircuit) -> Dict[str, int]:
        """获取门统计信息"""
        stats = {}
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            stats[gate_name] = stats.get(gate_name, 0) + 1
        return stats


def create_example_circuits() -> List[QuantumCircuit]:
    """创建一些示例电路用于测试"""
    circuits = []
    
    # 示例1: 简单的叠加态
    qc1 = QuantumCircuit(1, 1, name="superposition")
    qc1.h(0)
    qc1.measure(0, 0)
    circuits.append(qc1)
    
    # 示例2: 贝尔态
    qc2 = QuantumCircuit(2, 2, name="bell_state")
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.measure([0, 1], [0, 1])
    circuits.append(qc2)
    
    # 示例3: 复杂的多门电路
    qc3 = QuantumCircuit(3, 3, name="complex_circuit")
    qc3.h(0)
    qc3.rx(np.pi/4, 1)
    qc3.ry(np.pi/3, 2)
    qc3.cx(0, 1)
    qc3.cz(1, 2)
    qc3.rz(np.pi/6, 0)
    qc3.measure([0, 1, 2], [0, 1, 2])
    circuits.append(qc3)
    
    # 示例4: 包含T门的电路
    qc4 = QuantumCircuit(2, 2, name="t_gate_circuit")
    qc4.h(0)
    qc4.t(0)
    qc4.cx(0, 1)
    qc4.s(1)
    qc4.measure([0, 1], [0, 1])
    circuits.append(qc4)
    
    return circuits


def demo_ion_trap_compilation():
    """演示离子阱编译器的使用"""
    print("离子阱量子计算编译器演示")
    print("=" * 50)
    
    # 创建编译器
    compiler = IonTrapCompiler(optimization_level=1)
    
    # 获取示例电路
    circuits = create_example_circuits()
    
    # 编译每个电路
    for i, original_circuit in enumerate(circuits, 1):
        print(f"\n示例 {i}: {original_circuit.name}")
        print("-" * 30)
        
        print("原始电路:")
        print(original_circuit)
        
        # 编译电路
        ion_circuit = compiler.compile_circuit(original_circuit)
        
        print(f"\n编译后的离子阱电路:")
        print(ion_circuit)
        
        # 统计门的使用情况
        original_stats = compiler.get_gate_statistics(original_circuit)
        ion_stats = compiler.get_gate_statistics(ion_circuit)
        
        print(f"\n门统计对比:")
        print(f"原始电路: {original_stats}")
        print(f"离子阱电路: {ion_stats}")
        
        # 验证功能等价性（对于小电路）
        if original_circuit.num_qubits <= 3:
            verify_equivalence(original_circuit, ion_circuit)


def verify_equivalence(original: QuantumCircuit, compiled: QuantumCircuit):
    """验证编译前后电路的功能等价性"""
    try:
        # 移除测量操作进行比较
        orig_no_measure = original.remove_final_measurements(inplace=False)
        comp_no_measure = compiled.remove_final_measurements(inplace=False)
        
        # 使用状态向量模拟器比较
        from qiskit_aer import StatevectorSimulator
        
        simulator = StatevectorSimulator()
        
        # 运行原始电路
        orig_result = simulator.run(orig_no_measure).result()
        orig_state = orig_result.get_statevector()
        
        # 运行编译后的电路
        comp_result = simulator.run(comp_no_measure).result()
        comp_state = comp_result.get_statevector()
        
        # 计算保真度
        fidelity = abs(np.vdot(orig_state, comp_state)) ** 2
        
        print(f"电路保真度: {fidelity:.6f}")
        if fidelity > 0.999:
            print("✓ 编译结果正确")
        else:
            print("✗ 编译结果可能有误")
            
    except Exception as e:
        print(f"保真度验证失败: {e}")


if __name__ == "__main__":
    # 运行演示
    demo_ion_trap_compilation()
    
    print("\n" + "=" * 50)
    print("离子阱门详细信息:")
    print("=" * 50)
    
    # 展示各种离子阱门的详细信息
    print("\n1. MS门 (Mølmer-Sørensen gate)")
    print("   - 用途: 两量子比特纠缠操作")
    print("   - 实现: 通过激光与离子的相互作用")
    print("   - 参数: phi (相位), theta (旋转角度)")
    
    print("\n2. R_theta_phi门")
    print("   - 用途: 任意单量子比特旋转")
    print("   - 实现: 通过激光脉冲")
    print("   - 参数: theta (极角), phi (方位角)")
    
    print("\n3. Virtual Z门")
    print("   - 用途: Z旋转操作")
    print("   - 实现: 软件相位调整，无需激光")
    print("   - 参数: phi (相位角度)")
    print("   - 优势: 可与其他门合并优化") 