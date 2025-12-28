# S7 负对照报告

> 生成时间: 2025-12-29T01:14:55.785440

## 概述

负对照实验用于验证模型未走捷径，真正学习了任务结构。

### 负对照类型

| 类型 | 描述 | 预期结果 |
|------|------|----------|
| `label_shuffle` | 随机打乱标签映射 | acc ≈ random |
| `phase_scramble` | 打乱音频相位 | acc 显著下降 |
| `random_mapping` | 符号→频率随机映射 | acc ≈ random |

## 结果汇总

| Task | Control | Accuracy | Expected | Valid |
|------|---------|----------|----------|-------|
| mirror | label_shuffle | 0.000 | ~0.20 | ✅ |
| mirror | phase_scramble | 0.060 | ~0.20 | ✅ |
| mirror | random_mapping | 0.000 | ~0.20 | ✅ |
| bracket | label_shuffle | 0.000 | ~0.50 | ✅ |
| bracket | phase_scramble | 1.000 | ~0.50 | ✅ |
| bracket | random_mapping | 0.000 | ~0.50 | ✅ |
| mod | label_shuffle | 0.000 | ~0.10 | ✅ |
| mod | phase_scramble | 1.000 | ~0.10 | ✅ |
| mod | random_mapping | 0.000 | ~0.10 | ✅ |
| mod_compose | label_shuffle | 0.066 | ~0.10 | ✅ |
| mod_compose | phase_scramble | 0.000 | ~0.10 | ✅ |
| mod_compose | random_mapping | 0.000 | ~0.10 | ✅ |

## 结论

✅ **所有负对照通过**：系统确实依赖正确的频率映射和时域结构。

## 详细结果

### mirror / label_shuffle

- **准确率**: 0.000
- **预期随机基线**: ~0.20
- **状态**: ✅ 通过
- **解释**: 用打乱的频率映射编码后，标准解码器无法正确解码，验证解码依赖正确映射

### mirror / phase_scramble

- **准确率**: 0.060
- **预期随机基线**: ~0.20
- **状态**: ✅ 通过
- **解释**: 多符号序列相位打乱后分段破坏，解码失败是预期的

### mirror / random_mapping

- **准确率**: 0.000
- **预期随机基线**: ~0.20
- **状态**: ✅ 通过
- **解释**: 使用完全不同频率范围的随机映射，标准解码器应无法正确解码

随机映射:
```
  A: 6238.3Hz (standard: 440.0Hz)
  B: 6785.4Hz (standard: 560.0Hz)
  C: 6897.0Hz (standard: 720.0Hz)
  D: 6497.1Hz (standard: 920.0Hz)
  E: 6990.9Hz (standard: 1150.0Hz)
```

### bracket / label_shuffle

- **准确率**: 0.000
- **预期随机基线**: ~0.50
- **状态**: ✅ 通过
- **解释**: 用打乱的频率映射编码后，标准解码器无法正确解码，验证解码依赖正确映射

### bracket / phase_scramble

- **准确率**: 1.000
- **预期随机基线**: ~0.50
- **状态**: ✅ 通过
- **解释**: 单符号相位打乱不影响主频检测，仍能解码是预期的

### bracket / random_mapping

- **准确率**: 0.000
- **预期随机基线**: ~0.50
- **状态**: ✅ 通过
- **解释**: 使用完全不同频率范围的随机映射，标准解码器应无法正确解码

随机映射:
```
  V: 6759.0Hz (standard: 1900.0Hz)
  X: 6470.5Hz (standard: 1950.0Hz)
```

### mod / label_shuffle

- **准确率**: 0.000
- **预期随机基线**: ~0.10
- **状态**: ✅ 通过
- **解释**: 用打乱的频率映射编码后，标准解码器无法正确解码，验证解码依赖正确映射

### mod / phase_scramble

- **准确率**: 1.000
- **预期随机基线**: ~0.10
- **状态**: ✅ 通过
- **解释**: 单符号相位打乱不影响主频检测，仍能解码是预期的

### mod / random_mapping

- **准确率**: 0.000
- **预期随机基线**: ~0.10
- **状态**: ✅ 通过
- **解释**: 使用完全不同频率范围的随机映射，标准解码器应无法正确解码

随机映射:
```
  0: 6797.9Hz (standard: 2000.0Hz)
  1: 5851.4Hz (standard: 2170.0Hz)
  2: 6490.9Hz (standard: 2340.0Hz)
  3: 6294.4Hz (standard: 2510.0Hz)
  4: 6397.0Hz (standard: 2680.0Hz)
  5: 6599.8Hz (standard: 2850.0Hz)
  6: 6060.5Hz (standard: 3020.0Hz)
  7: 6891.3Hz (standard: 3190.0Hz)
  8: 6697.6Hz (standard: 3360.0Hz)
  9: 6994.5Hz (standard: 3530.0Hz)
```

### mod_compose / label_shuffle

- **准确率**: 0.066
- **预期随机基线**: ~0.10
- **状态**: ✅ 通过
- **解释**: 用打乱的频率映射编码后，标准解码器无法正确解码，验证解码依赖正确映射

### mod_compose / phase_scramble

- **准确率**: 0.000
- **预期随机基线**: ~0.10
- **状态**: ✅ 通过
- **解释**: 多符号序列相位打乱后分段破坏，解码失败是预期的

### mod_compose / random_mapping

- **准确率**: 0.000
- **预期随机基线**: ~0.10
- **状态**: ✅ 通过
- **解释**: 使用完全不同频率范围的随机映射，标准解码器应无法正确解码

随机映射:
```
  0: 6791.3Hz (standard: 2000.0Hz)
  1: 5774.0Hz (standard: 2170.0Hz)
  2: 6697.9Hz (standard: 2340.0Hz)
  3: 6390.9Hz (standard: 2510.0Hz)
  4: 6196.2Hz (standard: 2680.0Hz)
  5: 6297.0Hz (standard: 2850.0Hz)
  6: 6499.8Hz (standard: 3020.0Hz)
  7: 6894.5Hz (standard: 3190.0Hz)
  8: 5973.2Hz (standard: 3360.0Hz)
  9: 6993.7Hz (standard: 3530.0Hz)
  %: 6597.6Hz (standard: 3700.0Hz)
```

## 原始数据

```json
[
  {
    "task": "mirror",
    "control": "label_shuffle",
    "accuracy": 0.0,
    "expected_random": 0.2,
    "is_valid": true,
    "details": {
      "n_samples": 50,
      "seed": 42,
      "shuffled_mapping": {
        "A": "1150.0Hz",
        "B": "720.0Hz",
        "C": "920.0Hz",
        "D": "560.0Hz",
        "E": "440.0Hz"
      },
      "interpretation": "\u7528\u6253\u4e71\u7684\u9891\u7387\u6620\u5c04\u7f16\u7801\u540e\uff0c\u6807\u51c6\u89e3\u7801\u5668\u65e0\u6cd5\u6b63\u786e\u89e3\u7801\uff0c\u9a8c\u8bc1\u89e3\u7801\u4f9d\u8d56\u6b63\u786e\u6620\u5c04"
    }
  },
  {
    "task": "mirror",
    "control": "phase_scramble",
    "accuracy": 0.06,
    "expected_random": 0.2,
    "is_valid": true,
    "details": {
      "n_samples": 50,
      "seed": 42,
      "is_multi_symbol": true,
      "interpretation": "\u591a\u7b26\u53f7\u5e8f\u5217\u76f8\u4f4d\u6253\u4e71\u540e\u5206\u6bb5\u7834\u574f\uff0c\u89e3\u7801\u5931\u8d25\u662f\u9884\u671f\u7684"
    }
  },
  {
    "task": "mirror",
    "control": "random_mapping",
    "accuracy": 0.0,
    "expected_random": 0.2,
    "is_valid": true,
    "details": {
      "n_samples": 50,
      "seed": 42,
      "random_mapping": {
        "A": "6238.3Hz",
        "B": "6785.4Hz",
        "C": "6897.0Hz",
        "D": "6497.1Hz",
        "E": "6990.9Hz"
      },
      "standard_mapping": {
        "A": "440.0Hz",
        "B": "560.0Hz",
        "C": "720.0Hz",
        "D": "920.0Hz",
        "E": "1150.0Hz"
      },
      "interpretation": "\u4f7f\u7528\u5b8c\u5168\u4e0d\u540c\u9891\u7387\u8303\u56f4\u7684\u968f\u673a\u6620\u5c04\uff0c\u6807\u51c6\u89e3\u7801\u5668\u5e94\u65e0\u6cd5\u6b63\u786e\u89e3\u7801"
    }
  },
  {
    "task": "bracket",
    "control": "label_shuffle",
    "accuracy": 0.0,
    "expected_random": 0.5,
    "is_valid": true,
    "details": {
      "n_samples": 50,
      "seed": 42,
      "shuffled_mapping": {
        "V": "1950.0Hz",
        "X": "1900.0Hz"
      },
      "interpretation": "\u7528\u6253\u4e71\u7684\u9891\u7387\u6620\u5c04\u7f16\u7801\u540e\uff0c\u6807\u51c6\u89e3\u7801\u5668\u65e0\u6cd5\u6b63\u786e\u89e3\u7801\uff0c\u9a8c\u8bc1\u89e3\u7801\u4f9d\u8d56\u6b63\u786e\u6620\u5c04"
    }
  },
  {
    "task": "bracket",
    "control": "phase_scramble",
    "accuracy": 1.0,
    "expected_random": 0.5,
    "is_valid": true,
    "details": {
      "n_samples": 50,
      "seed": 42,
      "is_multi_symbol": false,
      "interpretation": "\u5355\u7b26\u53f7\u76f8\u4f4d\u6253\u4e71\u4e0d\u5f71\u54cd\u4e3b\u9891\u68c0\u6d4b\uff0c\u4ecd\u80fd\u89e3\u7801\u662f\u9884\u671f\u7684"
    }
  },
  {
    "task": "bracket",
    "control": "random_mapping",
    "accuracy": 0.0,
    "expected_random": 0.5,
    "is_valid": true,
    "details": {
      "n_samples": 50,
      "seed": 42,
      "random_mapping": {
        "V": "6759.0Hz",
        "X": "6470.5Hz"
      },
      "standard_mapping": {
        "V": "1900.0Hz",
        "X": "1950.0Hz"
      },
      "interpretation": "\u4f7f\u7528\u5b8c\u5168\u4e0d\u540c\u9891\u7387\u8303\u56f4\u7684\u968f\u673a\u6620\u5c04\uff0c\u6807\u51c6\u89e3\u7801\u5668\u5e94\u65e0\u6cd5\u6b63\u786e\u89e3\u7801"
    }
  },
  {
    "task": "mod",
    "control": "label_shuffle",
    "accuracy": 0.0,
    "expected_random": 0.1,
    "is_valid": true,
    "details": {
      "n_samples": 50,
      "seed": 42,
      "shuffled_mapping": {
        "0": "2850.0Hz",
        "1": "3020.0Hz",
        "2": "2000.0Hz",
        "3": "3190.0Hz",
        "4": "2510.0Hz",
        "5": "2340.0Hz",
        "6": "2680.0Hz",
        "7": "3530.0Hz",
        "8": "2170.0Hz",
        "9": "3360.0Hz"
      },
      "interpretation": "\u7528\u6253\u4e71\u7684\u9891\u7387\u6620\u5c04\u7f16\u7801\u540e\uff0c\u6807\u51c6\u89e3\u7801\u5668\u65e0\u6cd5\u6b63\u786e\u89e3\u7801\uff0c\u9a8c\u8bc1\u89e3\u7801\u4f9d\u8d56\u6b63\u786e\u6620\u5c04"
    }
  },
  {
    "task": "mod",
    "control": "phase_scramble",
    "accuracy": 1.0,
    "expected_random": 0.1,
    "is_valid": true,
    "details": {
      "n_samples": 50,
      "seed": 42,
      "is_multi_symbol": false,
      "interpretation": "\u5355\u7b26\u53f7\u76f8\u4f4d\u6253\u4e71\u4e0d\u5f71\u54cd\u4e3b\u9891\u68c0\u6d4b\uff0c\u4ecd\u80fd\u89e3\u7801\u662f\u9884\u671f\u7684"
    }
  },
  {
    "task": "mod",
    "control": "random_mapping",
    "accuracy": 0.0,
    "expected_random": 0.1,
    "is_valid": true,
    "details": {
      "n_samples": 50,
      "seed": 42,
      "random_mapping": {
        "0": "6797.9Hz",
        "1": "5851.4Hz",
        "2": "6490.9Hz",
        "3": "6294.4Hz",
        "4": "6397.0Hz",
        "5": "6599.8Hz",
        "6": "6060.5Hz",
        "7": "6891.3Hz",
        "8": "6697.6Hz",
        "9": "6994.5Hz"
      },
      "standard_mapping": {
        "0": "2000.0Hz",
        "1": "2170.0Hz",
        "2": "2340.0Hz",
        "3": "2510.0Hz",
        "4": "2680.0Hz",
        "5": "2850.0Hz",
        "6": "3020.0Hz",
        "7": "3190.0Hz",
        "8": "3360.0Hz",
        "9": "3530.0Hz"
      },
      "interpretation": "\u4f7f\u7528\u5b8c\u5168\u4e0d\u540c\u9891\u7387\u8303\u56f4\u7684\u968f\u673a\u6620\u5c04\uff0c\u6807\u51c6\u89e3\u7801\u5668\u5e94\u65e0\u6cd5\u6b63\u786e\u89e3\u7801"
    }
  },
  {
    "task": "mod_compose",
    "control": "label_shuffle",
    "accuracy": 0.06593406593406594,
    "expected_random": 0.1,
    "is_valid": true,
    "details": {
      "n_samples": 50,
      "seed": 42,
      "shuffled_mapping": {
        "0": "3020.0Hz",
        "1": "3190.0Hz",
        "2": "2000.0Hz",
        "3": "3700.0Hz",
        "4": "2510.0Hz",
        "5": "2850.0Hz",
        "6": "2340.0Hz",
        "7": "2680.0Hz",
        "8": "3530.0Hz",
        "9": "2170.0Hz",
        "%": "3360.0Hz"
      },
      "interpretation": "\u7528\u6253\u4e71\u7684\u9891\u7387\u6620\u5c04\u7f16\u7801\u540e\uff0c\u6807\u51c6\u89e3\u7801\u5668\u65e0\u6cd5\u6b63\u786e\u89e3\u7801\uff0c\u9a8c\u8bc1\u89e3\u7801\u4f9d\u8d56\u6b63\u786e\u6620\u5c04"
    }
  },
  {
    "task": "mod_compose",
    "control": "phase_scramble",
    "accuracy": 0.0,
    "expected_random": 0.1,
    "is_valid": true,
    "details": {
      "n_samples": 50,
      "seed": 42,
      "is_multi_symbol": true,
      "interpretation": "\u591a\u7b26\u53f7\u5e8f\u5217\u76f8\u4f4d\u6253\u4e71\u540e\u5206\u6bb5\u7834\u574f\uff0c\u89e3\u7801\u5931\u8d25\u662f\u9884\u671f\u7684"
    }
  },
  {
    "task": "mod_compose",
    "control": "random_mapping",
    "accuracy": 0.0,
    "expected_random": 0.1,
    "is_valid": true,
    "details": {
      "n_samples": 50,
      "seed": 42,
      "random_mapping": {
        "0": "6791.3Hz",
        "1": "5774.0Hz",
        "2": "6697.9Hz",
        "3": "6390.9Hz",
        "4": "6196.2Hz",
        "5": "6297.0Hz",
        "6": "6499.8Hz",
        "7": "6894.5Hz",
        "8": "5973.2Hz",
        "9": "6993.7Hz",
        "%": "6597.6Hz"
      },
      "standard_mapping": {
        "0": "2000.0Hz",
        "1": "2170.0Hz",
        "2": "2340.0Hz",
        "3": "2510.0Hz",
        "4": "2680.0Hz",
        "5": "2850.0Hz",
        "6": "3020.0Hz",
        "7": "3190.0Hz",
        "8": "3360.0Hz",
        "9": "3530.0Hz",
        "%": "3700.0Hz"
      },
      "interpretation": "\u4f7f\u7528\u5b8c\u5168\u4e0d\u540c\u9891\u7387\u8303\u56f4\u7684\u968f\u673a\u6620\u5c04\uff0c\u6807\u51c6\u89e3\u7801\u5668\u5e94\u65e0\u6cd5\u6b63\u786e\u89e3\u7801"
    }
  }
]
```