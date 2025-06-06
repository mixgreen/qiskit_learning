#!/usr/bin/env python3
"""
离子阱编译器正确性测试
====================

该模块用于验证离子阱编译器的正确性，通过比较原始电路和编译后电路的状态向量。
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix
from ion_trap_compiler import IonTrapCompiler
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IonTrapCompilerTester:
    """离子阱编译器测试类"""
    
    def __init__(self, tolerance: float = 1e-10):
        """
        初始化测试器
        
        Args:
            tolerance: 状态向量比较的数值容忍度
        """
        self.tolerance = tolerance
        self.compiler = IonTrapCompiler()
        self.simulator = AerSimulator(method='statevector')
        self.test_results = []
        
    def compare_states(self, state1: np.ndarray, state2: np.ndarray) -> Tuple[bool, float]:
        """
        比较两个量子态是否相等（考虑全局相位）
        
        Args:
            state1: 第一个状态向量
            state2: 第二个状态向量
            
        Returns:
            (是否相等, 保真度)
        """
        # 计算保真度
        fidelity = abs(np.vdot(state1, state2))**2
        
        # 考虑全局相位差异，计算最小距离
        # 尝试不同的全局相位
        phases = [1, -1, 1j, -1j]
        min_distance = float('inf')
        
        for phase in phases:
            diff = np.linalg.norm(state1 - phase * state2)
            min_distance = min(min_distance, diff)
            
        is_equal = min_distance < self.tolerance
        return is_equal, fidelity
    
    def run_circuit_test(self, original_circuit: QuantumCircuit, test_name: str) -> Dict:
        """
        运行单个电路测试
        
        Args:
            original_circuit: 原始量子电路
            test_name: 测试名称
            
        Returns:
            测试结果字典
        """
        logger.info(f"开始测试: {test_name}")
        
        try:
            # 编译电路
            compiled_circuit = self.compiler.compile_circuit(original_circuit)
            
            # 获取原始电路的状态向量
            original_sv = Statevector.from_instruction(original_circuit)
            original_state = original_sv.data
            
            # 获取编译后电路的状态向量
            compiled_sv = Statevector.from_instruction(compiled_circuit)
            compiled_state = compiled_sv.data
            
            # 比较状态
            is_equal, fidelity = self.compare_states(original_state, compiled_state)
            
            # 记录结果
            result = {
                'test_name': test_name,
                'passed': is_equal,
                'fidelity': fidelity,
                'original_depth': original_circuit.depth(),
                'compiled_depth': compiled_circuit.depth(),
                'original_gates': original_circuit.count_ops(),
                'compiled_gates': compiled_circuit.count_ops(),
                'original_state': original_state,
                'compiled_state': compiled_state
            }
            
            self.test_results.append(result)
            
            if is_equal:
                logger.info(f"✓ {test_name} 通过 (保真度: {fidelity:.10f})")
            else:
                logger.warning(f"✗ {test_name} 失败 (保真度: {fidelity:.10f})")
                
            return result
            
        except Exception as e:
            logger.error(f"✗ {test_name} 出错: {str(e)}")
            result = {
                'test_name': test_name,
                'passed': False,
                'error': str(e)
            }
            self.test_results.append(result)
            return result
    
    def test_single_qubit_gates(self) -> List[Dict]:
        """测试单量子比特门"""
        logger.info("=== 开始单量子比特门测试 ===")
        
        test_results = []
        
        # 基本门测试
        gates = [
            ('Hadamard', lambda qc, q: qc.h(q)),
            ('Pauli-X', lambda qc, q: qc.x(q)),
            ('Pauli-Y', lambda qc, q: qc.y(q)),
            ('Pauli-Z', lambda qc, q: qc.z(q)),
            ('S门', lambda qc, q: qc.s(q)),
            ('T门', lambda qc, q: qc.t(q)),
            ('S†门', lambda qc, q: qc.sdg(q)),
            ('T†门', lambda qc, q: qc.tdg(q)),
        ]
        
        for gate_name, gate_func in gates:
            qc = QuantumCircuit(1)
            gate_func(qc, 0)
            result = self.run_circuit_test(qc, f"单量子比特门 - {gate_name}")
            test_results.append(result)
        
        # 参数化旋转门测试
        angles = [np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, np.pi, 4*np.pi/3, 3*np.pi/2]
        
        for angle in angles:
            # RX门
            qc = QuantumCircuit(1)
            qc.rx(angle, 0)
            result = self.run_circuit_test(qc, f"RX门 (θ={angle:.3f})")
            test_results.append(result)
            
            # RY门
            qc = QuantumCircuit(1)
            qc.ry(angle, 0)
            result = self.run_circuit_test(qc, f"RY门 (θ={angle:.3f})")
            test_results.append(result)
            
            # RZ门
            qc = QuantumCircuit(1)
            qc.rz(angle, 0)
            result = self.run_circuit_test(qc, f"RZ门 (θ={angle:.3f})")
            test_results.append(result)
        
        return test_results
    
    def test_two_qubit_gates(self) -> List[Dict]:
        """测试双量子比特门"""
        logger.info("=== 开始双量子比特门测试 ===")
        
        test_results = []
        
        # CNOT门测试 - 不同的控制和目标量子比特组合
        cnot_configs = [(0, 1), (1, 0)]
        
        for control, target in cnot_configs:
            qc = QuantumCircuit(2)
            qc.cx(control, target)
            result = self.run_circuit_test(qc, f"CNOT门 (控制: {control}, 目标: {target})")
            test_results.append(result)
        
        # CZ门测试
        cz_configs = [(0, 1), (1, 0)]
        
        for q1, q2 in cz_configs:
            qc = QuantumCircuit(2)
            qc.cz(q1, q2)
            result = self.run_circuit_test(qc, f"CZ门 ({q1}, {q2})")
            test_results.append(result)
        
        return test_results
    
    def test_composite_circuits(self) -> List[Dict]:
        """测试复合电路"""
        logger.info("=== 开始复合电路测试 ===")
        
        test_results = []
        
        # Bell态制备
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        result = self.run_circuit_test(qc, "Bell态制备 |Φ+⟩")
        test_results.append(result)
        
        # GHZ态制备
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        result = self.run_circuit_test(qc, "GHZ态制备")
        test_results.append(result)
        
        # 量子傅里叶变换 (2量子比特)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cp(np.pi/2, 0, 1)
        qc.h(1)
        qc.swap(0, 1)
        result = self.run_circuit_test(qc, "2量子比特QFT")
        test_results.append(result)
        
        # 随机电路
        np.random.seed(42)  # 确保可重现性
        for i in range(5):
            qc = QuantumCircuit(2)
            for _ in range(10):
                gate_type = np.random.choice(['h', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'cx'])
                if gate_type in ['h', 'x', 'y', 'z']:
                    qubit = np.random.choice([0, 1])
                    getattr(qc, gate_type)(qubit)
                elif gate_type in ['rx', 'ry', 'rz']:
                    qubit = np.random.choice([0, 1])
                    angle = np.random.uniform(0, 2*np.pi)
                    getattr(qc, gate_type)(angle, qubit)
                elif gate_type == 'cx':
                    control, target = np.random.choice([0, 1], 2, replace=False)
                    qc.cx(control, target)
            
            result = self.run_circuit_test(qc, f"随机电路 #{i+1}")
            test_results.append(result)
        
        return test_results
    
    def test_superposition_and_entanglement(self) -> List[Dict]:
        """测试叠加态和纠缠态"""
        logger.info("=== 开始叠加态和纠缠态测试 ===")
        
        test_results = []
        
        # 不同角度的叠加态
        angles = [np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4]
        
        for angle in angles:
            qc = QuantumCircuit(1)
            qc.ry(2*angle, 0)  # |ψ⟩ = cos(θ)|0⟩ + sin(θ)|1⟩
            result = self.run_circuit_test(qc, f"叠加态 (θ={angle:.3f})")
            test_results.append(result)
        
        # 不同的Bell态
        bell_states = [
            ("Φ+", lambda qc: [qc.h(0), qc.cx(0, 1)]),
            ("Φ-", lambda qc: [qc.x(1), qc.h(0), qc.cx(0, 1)]),
            ("Ψ+", lambda qc: [qc.x(0), qc.h(0), qc.cx(0, 1)]),
            ("Ψ-", lambda qc: [qc.x(0), qc.x(1), qc.h(0), qc.cx(0, 1)]),
        ]
        
        for state_name, state_prep in bell_states:
            qc = QuantumCircuit(2)
            for gate in state_prep(qc):
                pass  # 门已经应用
            result = self.run_circuit_test(qc, f"Bell态 |{state_name}⟩")
            test_results.append(result)
        
        return test_results
    
    def run_all_tests(self) -> Dict:
        """运行所有测试"""
        logger.info("开始运行离子阱编译器正确性测试...")
        
        # 清空之前的结果
        self.test_results = []
        
        # 运行各类测试
        single_qubit_results = self.test_single_qubit_gates()
        two_qubit_results = self.test_two_qubit_gates()
        composite_results = self.test_composite_circuits()
        superposition_results = self.test_superposition_and_entanglement()
        
        # 统计结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.get('passed', False))
        failed_tests = total_tests - passed_tests
        
        # 计算平均保真度
        fidelities = [result.get('fidelity', 0) for result in self.test_results if 'fidelity' in result]
        avg_fidelity = np.mean(fidelities) if fidelities else 0
        min_fidelity = np.min(fidelities) if fidelities else 0
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'average_fidelity': avg_fidelity,
            'minimum_fidelity': min_fidelity,
            'test_results': self.test_results
        }
        
        # 打印总结
        logger.info("=" * 50)
        logger.info("测试总结:")
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过测试: {passed_tests}")
        logger.info(f"失败测试: {failed_tests}")
        logger.info(f"通过率: {summary['pass_rate']:.2%}")
        logger.info(f"平均保真度: {avg_fidelity:.10f}")
        logger.info(f"最低保真度: {min_fidelity:.10f}")
        logger.info("=" * 50)
        
        return summary
    
    def generate_test_report(self, filename: str = "ion_trap_compiler_test_report.txt"):
        """生成详细的测试报告"""
        if not self.test_results:
            logger.warning("没有测试结果可用于生成报告")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("离子阱编译器正确性测试报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 总体统计
            total_tests = len(self.test_results)
            passed_tests = sum(1 for result in self.test_results if result.get('passed', False))
            failed_tests = total_tests - passed_tests
            
            f.write("总体统计:\n")
            f.write(f"总测试数: {total_tests}\n")
            f.write(f"通过测试: {passed_tests}\n")
            f.write(f"失败测试: {failed_tests}\n")
            f.write(f"通过率: {passed_tests/total_tests:.2%}\n\n")
            
            # 详细结果
            f.write("详细测试结果:\n")
            f.write("-" * 30 + "\n")
            
            for i, result in enumerate(self.test_results, 1):
                f.write(f"{i:3d}. {result['test_name']}: ")
                if result.get('passed', False):
                    f.write(f"✓ 通过 (保真度: {result.get('fidelity', 0):.10f})\n")
                else:
                    if 'error' in result:
                        f.write(f"✗ 错误: {result['error']}\n")
                    else:
                        f.write(f"✗ 失败 (保真度: {result.get('fidelity', 0):.10f})\n")
                
                # 添加电路信息
                if 'original_depth' in result:
                    f.write(f"     原始电路深度: {result['original_depth']}, "
                           f"编译后深度: {result['compiled_depth']}\n")
                    f.write(f"     原始门数: {result.get('original_gates', {})}\n")
                    f.write(f"     编译后门数: {result.get('compiled_gates', {})}\n")
                f.write("\n")
        
        logger.info(f"测试报告已保存到: {filename}")
    
    def plot_fidelity_distribution(self):
        """绘制保真度分布图"""
        fidelities = [result.get('fidelity', 0) for result in self.test_results if 'fidelity' in result]
        
        if not fidelities:
            logger.warning("没有可用的保真度数据")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 保真度直方图
        plt.subplot(2, 2, 1)
        plt.hist(fidelities, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('保真度')
        plt.ylabel('频次')
        plt.title('保真度分布')
        plt.grid(True, alpha=0.3)
        
        # 保真度随测试编号变化
        plt.subplot(2, 2, 2)
        test_indices = range(1, len(fidelities) + 1)
        plt.plot(test_indices, fidelities, 'bo-', markersize=4)
        plt.xlabel('测试编号')
        plt.ylabel('保真度')
        plt.title('保真度变化趋势')
        plt.grid(True, alpha=0.3)
        
        # 低保真度测试分析
        plt.subplot(2, 2, 3)
        low_fidelity_threshold = 0.9999
        low_fidelity_tests = [(i, f) for i, f in enumerate(fidelities) if f < low_fidelity_threshold]
        
        if low_fidelity_tests:
            indices, values = zip(*low_fidelity_tests)
            plt.bar(range(len(indices)), values, alpha=0.7)
            plt.xlabel('低保真度测试')
            plt.ylabel('保真度')
            plt.title(f'保真度 < {low_fidelity_threshold} 的测试')
            plt.xticks(range(len(indices)), [f"Test {i+1}" for i in indices], rotation=45)
        else:
            plt.text(0.5, 0.5, '所有测试保真度均 > 0.9999', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('低保真度测试分析')
        
        # 统计信息
        plt.subplot(2, 2, 4)
        stats_text = f"""
        统计信息:
        
        总测试数: {len(fidelities)}
        平均保真度: {np.mean(fidelities):.10f}
        最小保真度: {np.min(fidelities):.10f}
        最大保真度: {np.max(fidelities):.10f}
        标准差: {np.std(fidelities):.2e}
        
        保真度 > 0.99999: {sum(f > 0.99999 for f in fidelities)}
        保真度 > 0.9999: {sum(f > 0.9999 for f in fidelities)}
        保真度 > 0.999: {sum(f > 0.999 for f in fidelities)}
        """
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('ion_trap_compiler_fidelity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("保真度分析图已保存为 ion_trap_compiler_fidelity_analysis.png")


def main():
    """主函数 - 运行完整的测试套件"""
    print("离子阱编译器正确性测试")
    print("=" * 50)
    
    # 创建测试器
    tester = IonTrapCompilerTester(tolerance=1e-10)
    
    # 运行所有测试
    summary = tester.run_all_tests()
    
    # 生成报告
    tester.generate_test_report()
    
    # 绘制分析图
    tester.plot_fidelity_distribution()
    
    # 显示失败的测试
    failed_tests = [result for result in tester.test_results if not result.get('passed', False)]
    if failed_tests:
        print("\n失败的测试:")
        print("-" * 30)
        for result in failed_tests:
            print(f"• {result['test_name']}")
            if 'error' in result:
                print(f"  错误: {result['error']}")
            elif 'fidelity' in result:
                print(f"  保真度: {result['fidelity']:.10f}")
    
    return summary


if __name__ == "__main__":
    main() 