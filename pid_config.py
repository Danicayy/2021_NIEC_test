"""
PID控制器参数配置文件
提供不同场景下的PID参数预设
"""

# PID参数预设配置
PID_PRESETS = {
    # 稳定型 - 适合初学者或要求稳定性的场景
    "stable": {
        "kp": 1.5,
        "ki": 0.05,
        "kd": 0.8,
        "description": "稳定型：响应平缓，适合初学者"
    },
    
    # 平衡型 - 默认推荐配置
    "balanced": {
        "kp": 2.0,
        "ki": 0.1,
        "kd": 0.5,
        "description": "平衡型：速度与稳定性的平衡"
    },
    
    # 快速型 - 适合高速循迹
    "fast": {
        "kp": 2.8,
        "ki": 0.15,
        "kd": 0.3,
        "description": "快速型：响应迅速，适合高速循迹"
    },
    
    # 精确型 - 适合复杂路径
    "precise": {
        "kp": 2.2,
        "ki": 0.2,
        "kd": 0.7,
        "description": "精确型：高精度跟踪，适合复杂路径"
    },
    
    # 抗干扰型 - 适合噪声环境
    "robust": {
        "kp": 1.8,
        "ki": 0.05,
        "kd": 1.0,
        "description": "抗干扰型：强抗噪能力，适合光照变化环境"
    }
}

def get_preset(preset_name):
    """
    获取预设参数
    
    Args:
        preset_name: 预设名称 ('stable', 'balanced', 'fast', 'precise', 'robust')
        
    Returns:
        dict: 包含kp, ki, kd参数的字典
    """
    if preset_name in PID_PRESETS:
        return PID_PRESETS[preset_name]
    else:
        print(f"未找到预设 '{preset_name}'，使用默认 'balanced' 配置")
        return PID_PRESETS["balanced"]

def list_presets():
    """列出所有可用的预设配置"""
    print("可用的PID预设配置:")
    for name, config in PID_PRESETS.items():
        print(f"  {name}: {config['description']}")
        print(f"    Kp={config['kp']}, Ki={config['ki']}, Kd={config['kd']}")
        print()

if __name__ == "__main__":
    # 测试代码
    list_presets()
