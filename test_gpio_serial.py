#!/usr/bin/env python3
"""
GPIO串口通信测试脚本
用于验证 /dev/serial0 (GPIO串口) 是否正常工作
"""

import serial
import time
import sys

def test_gpio_serial():
    """测试GPIO串口通信"""
    
    # 串口配置
    SERIAL_PORT = '/dev/serial0'
    BAUDRATE = 9600
    
    print("GPIO串口通信测试")
    print("=" * 50)
    print(f"串口: {SERIAL_PORT}")
    print(f"波特率: {BAUDRATE}")
    print("=" * 50)
    
    try:
        # 初始化串口
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUDRATE,
            timeout=1.0,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )
        
        # 清空缓冲区
        ser.flush()
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        print("✓ GPIO串口初始化成功")
        print(f"  端口信息: {ser.name}")
        print(f"  是否打开: {ser.is_open}")
        print(f"  波特率: {ser.baudrate}")
        print(f"  数据位: {ser.bytesize}")
        print(f"  停止位: {ser.stopbits}")
        print(f"  校验位: {ser.parity}")
        print()
        
        # 测试发送数据
        test_commands = [
            ("直走指令", 300001),
            ("停止指令", 300002),
            ("左转指令", "LEFT"),
            ("右转指令", "RIGHT"),
            ("测试offset", 25),
            ("红线丢失", None)
        ]
        
        print("开始测试发送数据...")
        print("-" * 30)
        
        for desc, cmd in test_commands:
            print(f"测试: {desc}")
            
            if isinstance(cmd, int):
                # 数字指令
                if cmd == 300001 or cmd == 300002:
                    # 控制指令
                    cmd_str = f"{cmd:06d}"
                    hex_data = bytes([int(digit) for digit in cmd_str])
                    print(f"  发送控制指令: {cmd} -> {list(hex_data)}")
                else:
                    # offset数据
                    head = 1 if cmd >= 0 else 0
                    abs_offset = abs(cmd)
                    tens_digit = abs_offset // 10
                    ones_digit = abs_offset % 10
                    hex_data = bytes([head, 0x00, 0x00, 0x00, tens_digit, ones_digit])
                    print(f"  发送offset数据: {cmd} -> {list(hex_data)}")
                
                ser.write(hex_data)
                
            elif isinstance(cmd, str):
                # 转弯指令
                if cmd == "LEFT":
                    hex_data = bytes([0x02, 0x00, 0x00, 0x00, 0x00, 0x02])
                elif cmd == "RIGHT":
                    hex_data = bytes([0x02, 0x00, 0x00, 0x00, 0x00, 0x01])
                
                print(f"  发送转弯指令: {cmd} -> {list(hex_data)}")
                ser.write(hex_data)
                
            elif cmd is None:
                # 红线丢失
                hex_data = bytes([0x66, 0x66, 0x66, 0x66, 0x66, 0x66])
                print(f"  发送红线丢失信号: -> {list(hex_data)}")
                ser.write(hex_data)
            
            # 等待并检查是否有回传数据
            time.sleep(0.1)
            if ser.in_waiting > 0:
                response = ser.read(ser.in_waiting)
                print(f"  收到回传: {response}")
            else:
                print(f"  无回传数据")
            
            print()
            time.sleep(0.5)
        
        print("✓ 数据发送测试完成")
        print()
        
        # 持续监听模式
        print("进入持续监听模式 (按Ctrl+C退出)...")
        print("等待下位机回传数据...")
        
        try:
            while True:
                if ser.in_waiting > 0:
                    data = ser.read(ser.in_waiting)
                    try:
                        decoded = data.decode('utf-8', errors='ignore').strip()
                        if decoded:
                            print(f"[{time.strftime('%H:%M:%S')}] 收到: {decoded}")
                    except:
                        print(f"[{time.strftime('%H:%M:%S')}] 收到原始数据: {list(data)}")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n用户中断，退出监听模式")
        
        # 关闭串口
        ser.close()
        print("✓ 串口已关闭")
        
    except serial.SerialException as e:
        print(f"✗ 串口错误: {e}")
        print("\n可能的解决方案:")
        print("1. 检查串口设备是否存在: ls -l /dev/serial*")
        print("2. 检查串口权限: sudo usermod -a -G dialout $USER")
        print("3. 检查GPIO串口是否正确配置")
        print("4. 重启系统后再试")
        
    except Exception as e:
        print(f"✗ 未知错误: {e}")

def check_serial_devices():
    """检查可用的串口设备"""
    import glob
    
    print("检查系统串口设备...")
    print("-" * 30)
    
    # 检查常见的串口设备
    patterns = [
        '/dev/ttyAMA*',
        '/dev/ttyUSB*', 
        '/dev/ttyACM*',
        '/dev/serial*'
    ]
    
    found_devices = []
    for pattern in patterns:
        devices = glob.glob(pattern)
        found_devices.extend(devices)
    
    if found_devices:
        print("发现的串口设备:")
        for device in sorted(found_devices):
            try:
                # 尝试获取设备信息
                import os
                if os.path.exists(device):
                    stat_info = os.stat(device)
                    print(f"  {device} (权限: {oct(stat_info.st_mode)[-3:]})")
                else:
                    print(f"  {device} (链接)")
            except:
                print(f"  {device}")
    else:
        print("未发现串口设备")
    
    print()

if __name__ == "__main__":
    print("GPIO串口测试工具")
    print("=" * 50)
    
    # 检查串口设备
    check_serial_devices()
    
    # 测试GPIO串口
    test_gpio_serial()
