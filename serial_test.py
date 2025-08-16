import serial
import time

SERIAL_PORT = '/dev/ttyUSB0'  # 替换成你的树莓派串口号（如COM3等）
BAUDRATE = 9600

def main():
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print(f"已打开串口 {SERIAL_PORT}，波特率{BAUDRATE}")

    try:
        while True:
            # 发送测试字符串
            test_data = "300001"
            ser.write(test_data.encode('utf-8'))
            print(f"发送: {test_data.strip()}")
            
            # 等待下位机回显
            time.sleep(0.5)
            if ser.in_waiting:
                recv_data = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"收到: {recv_data}")
            else:
                print("未收到下位机数据")
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("测试结束")
    finally:
        ser.close()

if __name__ == '__main__':
    main()