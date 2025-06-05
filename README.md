# Qiskit 学习项目

欢迎来到 Qiskit 学习项目！这个项目使用 `uv` 包管理器创建，旨在帮助你学习量子计算、Qiskit 框架和 OpenQASM 3.0 量子汇编语言。

## 🚀 快速开始

### 环境要求

- Python 3.8+
- uv 包管理器

### 项目设置

1. **激活虚拟环境**：

   ```bash
   # Windows (PowerShell)
   .venv\Scripts\activate

   # Linux/macOS
   source .venv/bin/activate
   ```
2. **运行主程序**：

   ```bash
   uv run python main.py
   ```
3. **启动 Jupyter Notebook**：

   ```bash
   uv run jupyter notebook
   ```

## 📦 已安装的包

- **qiskit**: 量子计算核心框架
- **qiskit-aer**: 量子电路模拟器
- **openqasm3**: OpenQASM 3.0 支持
- **matplotlib**: 数据可视化
- **jupyter**: 交互式开发环境
- **notebook**: Jupyter Notebook 支持

## 📁 项目结构

```
qiskit-learning-project/
├── main.py                          # 主程序，包含基本示例
├── examples/                        # 学习示例目录
│   ├── basic_gates.py              # 基本量子门示例
│   ├── openqasm3_examples.py       # OpenQASM 3.0 示例
│   ├── ion_trap_compiler.py        # 离子阱编译器核心代码
│   ├── ion_trap_usage_example.py   # 离子阱编译器使用示例
│   ├── ion_trap_measurement_demo.py # 离子阱测量特性演示
│   ├── qiskit_tutorial.ipynb       # Qiskit 基础教程
│   ├── openqasm3_tutorial.ipynb    # OpenQASM 3.0 教程
│   ├── ion_trap_tutorial.ipynb     # 离子阱编译器教程
│   └── bell_state.qasm             # OpenQASM 3.0 文件示例
├── .venv/                          # 虚拟环境
├── pyproject.toml                  # 项目配置
└── README.md                       # 项目说明
```

## 🎯 学习内容

### 1. 基础概念

- 量子比特和量子态
- 量子叠加和量子纠缠
- 量子测量

### 2. 量子门

- 泡利门 (X, Y, Z)
- Hadamard 门
- CNOT 门
- 旋转门 (RX, RY, RZ)

### 3. 量子电路

- 电路构建
- 电路模拟
- 结果分析

### 4. OpenQASM 3.0 🆕

- OpenQASM 3.0 语法和结构
- 与 Qiskit 的双向转换
- 变量和函数使用
- 文件操作和导入导出

### 5. 离子阱量子计算编译器 🚀

- 离子阱门集：MS门、R_theta_phi门、Virtual Z门
- 标准量子门到离子阱门的转换
- 电路优化和性能分析
- 自定义硬件后端支持

## 📚 使用方法

### 运行基本示例

```bash
# 运行主程序
uv run python main.py

# 运行量子门示例
uv run python examples/basic_gates.py

# 运行 OpenQASM 3.0 示例
uv run python examples/openqasm3_examples.py

# 运行离子阱编译器示例
uv run python examples/ion_trap_compiler.py

# 运行离子阱使用示例
uv run python examples/ion_trap_usage_example.py

# 运行离子阱测量特性演示
uv run python examples/ion_trap_measurement_demo.py
```

### 使用 Jupyter Notebook

```bash
# 启动 Jupyter
uv run jupyter notebook

# 推荐学习顺序：
# 1. examples/qiskit_tutorial.ipynb - Qiskit 基础
# 2. examples/openqasm3_tutorial.ipynb - OpenQASM 3.0
```

## 🔧 常用命令

```bash
# 添加新的包
uv add package_name

# 查看已安装的包
uv pip list

# 更新包
uv sync

# 运行 Python 脚本
uv run python script.py
```

## 🌟 OpenQASM 3.0 特色功能

OpenQASM 3.0 是本项目的重点学习内容，它提供了：

### 基本语法示例

```qasm
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];
measure q -> c;
```

### 高级特性

- **变量定义**: `angle theta = pi/4;`
- **条件执行**: `if (c == 1) { ... }`
- **自定义门**: 定义可重用的量子门
- **模块化**: 包含外部文件

### 与 Qiskit 的集成

```python
from qiskit.qasm3 import loads, dumps

# Qiskit 电路转换为 OpenQASM 3.0
qasm_code = dumps(quantum_circuit)

# OpenQASM 3.0 代码转换为 Qiskit 电路
quantum_circuit = loads(qasm_code)
```

## 🔬 离子阱编译器特色功能

### 离子阱门集介绍

**MS门 (Mølmer-Sørensen Gate)**
```python
ms_gate = MSGate(phi=0.0, theta=np.pi/2)
circuit.append(ms_gate, [qubit1, qubit2])
```

**R_theta_phi门 (单量子比特旋转门)**
```python
r_gate = RThetaPhiGate(theta=np.pi/2, phi=np.pi/4)
circuit.append(r_gate, [qubit])
```

**Virtual Z门 (虚拟Z门)**
```python
vz_gate = VirtualZGate(phi=np.pi/6)
circuit.append(vz_gate, [qubit])
```

### 编译器使用示例

```python
from examples.ion_trap_compiler import IonTrapCompiler

# 创建编译器
compiler = IonTrapCompiler(optimization_level=1)

# 编译标准电路为离子阱电路
standard_circuit = QuantumCircuit(2, 2)
standard_circuit.h(0)
standard_circuit.cx(0, 1)

ion_trap_circuit = compiler.compile_circuit(standard_circuit)

# 分析编译结果
stats = compiler.get_gate_statistics(ion_trap_circuit)
print(f"离子阱门统计: {stats}")
```

## 📖 学习资源

- [Qiskit 官方文档](https://qiskit.org/documentation/)
- [OpenQASM 3.0 规范](https://openqasm.com/)
- [Qiskit 教程](https://qiskit.org/learn/)
- [量子计算入门](https://qiskit.org/textbook/)

## 🎉 学习路径建议

1. **基础入门**

   - 运行 `main.py` 查看基本示例
   - 学习 `examples/basic_gates.py` 中的量子门
2. **深入学习**

   - 打开 `examples/qiskit_tutorial.ipynb` 进行交互式学习
   - 探索 `examples/openqasm3_examples.py` 中的 OpenQASM 3.0 示例
3. **高级应用**

   - 学习 `examples/openqasm3_tutorial.ipynb`
   - 探索 `examples/ion_trap_compiler.py` 中的离子阱编译器
   - 运行 `examples/ion_trap_usage_example.py` 学习编译器使用

4. **专业开发**

   - 学习 `examples/ion_trap_tutorial.ipynb`
   - 创建自己的 `.qasm` 文件
   - 定制离子阱编译器以适应特定硬件

5. **实践项目**

   - 实现经典量子算法并编译为离子阱电路
   - 使用 OpenQASM 3.0 编写复杂电路
   - 开发针对离子阱系统的优化算法

## 📝 注意事项

- 确保已正确激活虚拟环境
- 如果遇到包依赖问题，运行 `uv sync` 重新同步
- Jupyter Notebook 中的可视化需要正确配置 matplotlib
- OpenQASM 3.0 的某些高级特性可能需要最新版本的 Qiskit

## 🤝 贡献

欢迎提交问题和改进建议！

祝你学习愉快！🌟
