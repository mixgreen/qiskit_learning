#!/usr/bin/env python3
"""
离子阱编译器正确性验证框架
===============================

该模块用于验证离子阱编译器的正确性，通过比较原始电路和编译后电路的状态向量。
基于Mølmer-Sørensen门理论和IonQ的标准分解方法。

参考文献:
- Mølmer, K. & Sørensen, A. Phys. Rev. Lett. 82, 1971–1974 (1999)
- IonQ Native Gates Documentation
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_learning.ion_trap.compiler import IonTrapCompiler
from typing import List, Tuple, Dict, Optional
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IonTrapVerificationFramework:
    """离子阱编译器验证框架"""
    
    def __init__(self, tolerance: float = 1e-8):
        """
        初始化验证框架
        
        Args:
            tolerance: 状态比较的数值容忍度
        """
        self.tolerance = tolerance
        self.compiler = IonTrapCompiler()
        self.simulator = AerSimulator(method='statevector')
        self.test_results = []
        
    def compare_states(self, state1: np.ndarray, state2: np.ndarray) -> Tuple[bool, float, float]:
        """
        比较两个量子态，考虑全局相位
        
        Args:
            state1, state2: 要比较的状态向量
            
        Returns:
            (是否相同, 保真度, 最小距离)
        """
        # 计算保真度
        fidelity = abs(np.vdot(state1, state2))**2
        
        # 尝试不同全局相位
        phases = [1, -1, 1j, -1j, np.exp(1j*np.pi/4), np.exp(-1j*np.pi/4)]
        min_distance = float('inf')
        
        for phase in phases:
            diff = np.linalg.norm(state1 - phase * state2)
            min_distance = min(min_distance, diff)
        
        is_same = min_distance < self.tolerance
        return is_same, fidelity, min_distance
    
    def run_circuit(self, circuit: QuantumCircuit) -> np.ndarray:
        """
        运行量子电路并获取状态向量
        
        Args:
            circuit: 要运行的量子电路
            
        Returns:
            状态向量
        """
        try:
            # 使用Statevector直接计算
            statevector = Statevector.from_instruction(circuit)
            return statevector.data
        except Exception as e:
            logger.error(f"电路运行失败: {e}")
            # 备选方案：使用模拟器
            job = self.simulator.run(circuit)
            result = job.result()
            statevector = result.get_statevector()
            return statevector
    
    def test_single_case(self, test_name: str, circuit_func, num_qubits: int) -> Dict:
        """
        测试单个案例
        
        Args:
            test_name: 测试名称
            circuit_func: 创建电路的函数
            num_qubits: 量子比特数量
            
        Returns:
            测试结果字典
        """
        logger.info(f"测试案例: {test_name}")
        
        try:
            start_time = time.time()
            
            # 创建原始电路
            original_circuit = circuit_func()
            
            # 编译电路
            compiled_circuit = self.compiler.compile(original_circuit)
            
            # 运行两个电路
            original_state = self.run_circuit(original_circuit)
            compiled_state = self.run_circuit(compiled_circuit)
            
            # 比较状态
            is_same, fidelity, distance = self.compare_states(original_state, compiled_state)
            
            # 计算门数量和深度
            original_gates = original_circuit.count_ops()
            compiled_gates = compiled_circuit.count_ops()
            
            test_time = time.time() - start_time
            
            result = {
                'test_name': test_name,
                'passed': is_same,
                'fidelity': fidelity,
                'distance': distance,
                'original_gates': original_gates,
                'compiled_gates': compiled_gates,
                'original_depth': original_circuit.depth(),
                'compiled_depth': compiled_circuit.depth(),
                'test_time': test_time,
                'error_message': None
            }
            
            if is_same:
                logger.info(f"✓ {test_name} - 通过 (保真度: {fidelity:.6f})")
            else:
                logger.warning(f"✗ {test_name} - 失败 (保真度: {fidelity:.6f}, 距离: {distance:.6f})")
                
            return result
            
        except Exception as e:
            logger.error(f"✗ {test_name} - 错误: {e}")
            return {
                'test_name': test_name,
                'passed': False,
                'fidelity': 0.0,
                'distance': float('inf'),
                'error_message': str(e)
            }
    
    def create_test_circuits(self) -> List[Tuple[str, callable, int]]:
        """创建测试电路集合"""
        
        test_cases = []
        
        # === 单量子比特门测试 ===
        def create_x_circuit():
            qc = QuantumCircuit(1)
            qc.x(0)
            return qc
        test_cases.append(("X门", create_x_circuit, 1))
        
        def create_y_circuit():
            qc = QuantumCircuit(1)
            qc.y(0)
            return qc
        test_cases.append(("Y门", create_y_circuit, 1))
        
        def create_z_circuit():
            qc = QuantumCircuit(1)
            qc.z(0)
            return qc
        test_cases.append(("Z门", create_z_circuit, 1))
        
        def create_h_circuit():
            qc = QuantumCircuit(1)
            qc.h(0)
            return qc
        test_cases.append(("Hadamard门", create_h_circuit, 1))
        
        def create_s_circuit():
            qc = QuantumCircuit(1)
            qc.s(0)
            return qc
        test_cases.append(("S门", create_s_circuit, 1))
        
        def create_t_circuit():
            qc = QuantumCircuit(1)
            qc.t(0)
            return qc
        test_cases.append(("T门", create_t_circuit, 1))
        
        # === 参数化旋转门测试 ===
        def create_rx_circuit():
            qc = QuantumCircuit(1)
            qc.rx(np.pi/3, 0)
            return qc
        test_cases.append(("RX(π/3)门", create_rx_circuit, 1))
        
        def create_ry_circuit():
            qc = QuantumCircuit(1)
            qc.ry(np.pi/4, 0)
            return qc
        test_cases.append(("RY(π/4)门", create_ry_circuit, 1))
        
        def create_rz_circuit():
            qc = QuantumCircuit(1)
            qc.rz(np.pi/6, 0)
            return qc
        test_cases.append(("RZ(π/6)门", create_rz_circuit, 1))
        
        # === 双量子比特门测试 ===
        def create_cnot_circuit():
            qc = QuantumCircuit(2)
            qc.cx(0, 1)
            return qc
        test_cases.append(("CNOT门", create_cnot_circuit, 2))
        
        def create_cz_circuit():
            qc = QuantumCircuit(2)
            qc.cz(0, 1)
            return qc
        test_cases.append(("CZ门", create_cz_circuit, 2))
        
        # === 复合电路测试 ===
        def create_bell_circuit():
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            return qc
        test_cases.append(("Bell态制备", create_bell_circuit, 2))
        
        def create_ghz_circuit():
            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(1, 2)
            return qc
        test_cases.append(("GHZ态制备", create_ghz_circuit, 3))
        
        def create_quantum_fourier_circuit():
            qc = QuantumCircuit(3)
            # 简化的QFT
            qc.h(0)
            qc.cu1(np.pi/2, 0, 1)
            qc.cu1(np.pi/4, 0, 2)
            qc.h(1)
            qc.cu1(np.pi/2, 1, 2)
            qc.h(2)
            qc.swap(0, 2)
            return qc
        test_cases.append(("量子傅里叶变换", create_quantum_fourier_circuit, 3))
        
        # === 随机电路测试 ===
        def create_random_circuit_1():
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.ry(np.pi/7, 1)
            qc.cx(0, 1)
            qc.rz(np.pi/5, 0)
            qc.x(1)
            return qc
        test_cases.append(("随机电路1", create_random_circuit_1, 2))
        
        def create_random_circuit_2():
            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            qc.ry(np.pi/3, 2)
            qc.cz(1, 2)
            qc.rx(np.pi/8, 0)
            qc.s(1)
            return qc
        test_cases.append(("随机电路2", create_random_circuit_2, 3))
        
        # === 边界情况测试 ===
        def create_identity_circuit():
            qc = QuantumCircuit(2)
            # 恒等电路（什么都不做）
            return qc
        test_cases.append(("恒等电路", create_identity_circuit, 2))
        
        def create_all_gates_circuit():
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.x(1)
            qc.y(0)
            qc.z(1)
            qc.s(0)
            qc.t(1)
            qc.cx(0, 1)
            qc.cz(1, 0)
            return qc
        test_cases.append(("所有门类型", create_all_gates_circuit, 2))
        
        return test_cases
    
    def run_all_tests(self) -> Dict:
        """运行所有测试"""
        logger.info("开始离子阱编译器验证测试")
        logger.info("=" * 50)
        
        test_cases = self.create_test_circuits()
        results = []
        
        for test_name, circuit_func, num_qubits in test_cases:
            result = self.test_single_case(test_name, circuit_func, num_qubits)
            results.append(result)
            self.test_results.append(result)
        
        # 统计结果
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['passed'])
        failed_tests = total_tests - passed_tests
        
        # 计算平均保真度
        fidelities = [r['fidelity'] for r in results if 'fidelity' in r]
        avg_fidelity = np.mean(fidelities) if fidelities else 0.0
        min_fidelity = np.min(fidelities) if fidelities else 0.0
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests * 100,
            'average_fidelity': avg_fidelity,
            'minimum_fidelity': min_fidelity,
            'results': results
        }
        
        # 打印总结
        logger.info("=" * 50)
        logger.info("测试总结:")
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过测试: {passed_tests}")
        logger.info(f"失败测试: {failed_tests}")
        logger.info(f"成功率: {summary['success_rate']:.1f}%")
        logger.info(f"平均保真度: {avg_fidelity:.6f}")
        logger.info(f"最低保真度: {min_fidelity:.6f}")
        
        if failed_tests > 0:
            logger.info("\n失败的测试:")
            for result in results:
                if not result['passed']:
                    error_msg = result.get('error_message', f"距离: {result.get('distance', 'N/A')}")
                    logger.info(f"  - {result['test_name']}: {error_msg}")
        
        return summary
    
    def generate_report(self, filename: str = "verification_report.txt"):
        """生成详细的验证报告"""
        if not self.test_results:
            logger.warning("没有测试结果可以生成报告")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("离子阱编译器验证报告\n")
            f.write("=" * 50 + "\n\n")
            
            total_tests = len(self.test_results)
            passed_tests = sum(1 for r in self.test_results if r['passed'])
            
            f.write(f"测试概况:\n")
            f.write(f"  总测试数: {total_tests}\n")
            f.write(f"  通过测试: {passed_tests}\n")
            f.write(f"  成功率: {passed_tests/total_tests*100:.1f}%\n\n")
            
            f.write("详细测试结果:\n")
            f.write("-" * 50 + "\n")
            
            for result in self.test_results:
                f.write(f"\n测试名称: {result['test_name']}\n")
                f.write(f"  状态: {'通过' if result['passed'] else '失败'}\n")
                f.write(f"  保真度: {result.get('fidelity', 'N/A'):.6f}\n")
                f.write(f"  距离: {result.get('distance', 'N/A'):.6f}\n")
                if 'original_gates' in result:
                    f.write(f"  原始门数: {result['original_gates']}\n")
                    f.write(f"  编译后门数: {result['compiled_gates']}\n")
                    f.write(f"  原始深度: {result['original_depth']}\n")
                    f.write(f"  编译后深度: {result['compiled_depth']}\n")
                if result.get('error_message'):
                    f.write(f"  错误信息: {result['error_message']}\n")
        
        logger.info(f"验证报告已保存到: {filename}")
    
    def plot_results(self, save_path: str = "verification_results.png"):
        """绘制测试结果图表"""
        if not self.test_results:
            logger.warning("没有测试结果可以绘制")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 子图1: 成功率统计
        passed = sum(1 for r in self.test_results if r['passed'])
        failed = len(self.test_results) - passed
        
        ax1.pie([passed, failed], labels=['通过', '失败'], autopct='%1.1f%%',
                colors=['green', 'red'], startangle=90)
        ax1.set_title('测试结果统计')
        
        # 子图2: 保真度分布
        fidelities = [r['fidelity'] for r in self.test_results if 'fidelity' in r]
        test_names = [r['test_name'] for r in self.test_results if 'fidelity' in r]
        
        colors = ['green' if f > 0.99 else 'orange' if f > 0.95 else 'red' for f in fidelities]
        bars = ax2.bar(range(len(fidelities)), fidelities, color=colors)
        ax2.set_xlabel('测试案例')
        ax2.set_ylabel('保真度')
        ax2.set_title('各测试案例保真度')
        ax2.set_xticks(range(len(test_names)))
        ax2.set_xticklabels(test_names, rotation=45, ha='right')
        ax2.set_ylim(0, 1.1)
        ax2.axhline(y=0.99, color='black', linestyle='--', alpha=0.5, label='99%阈值')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"结果图表已保存到: {save_path}")

def main():
    """主函数"""
    print("离子阱编译器正确性验证")
    print("=" * 40)
    
    # 创建验证框架
    verifier = IonTrapVerificationFramework(tolerance=1e-8)
    
    # 运行所有测试
    summary = verifier.run_all_tests()
    
    # 生成报告和图表
    verifier.generate_report()
    verifier.plot_results()
    
    print(f"\n验证完成！成功率: {summary['success_rate']:.1f}%")
    return summary

if __name__ == "__main__":
    main() 