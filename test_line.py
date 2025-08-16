import cv2
import numpy as np
import serial
import time


CAMERA_INDEX = 0
FRAME_WIDTH = 160
FRAME_HEIGHT = 120
SERIAL_PORT = '/dev/ttyUSB0'  # 替换成你的树莓派串口号（如COM3等）
BAUDRATE = 115200

# 红色HSV阈值（你可以根据实际环境调整）
LOWER_RED1 = np.array([0, 100, 100])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([160, 100, 100])
UPPER_RED2 = np.array([179, 255, 255])

def get_red_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = cv2.bitwise_or(mask1, mask2)
    return mask

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.00)
    print(f"已打开串口 {SERIAL_PORT}，波特率{BAUDRATE}")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed!")
            continue

        # 只取下方1/3区域分析
        roi_y1 = int(FRAME_HEIGHT * 0.6)
        roi = frame[roi_y1:, :].copy()

        # 红色掩膜
        mask = get_red_mask(roi)
        mask_color = cv2.merge([mask, mask, mask])

        # 找红线重心
        moments = cv2.moments(mask)
        line_center = None
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            offset = cx - FRAME_WIDTH // 2

            # 1. 格式化为6位数字。正数前面补+，负数补-
            data_packet = f"{offset:06d}"  # 例如 +00035、-00123

            ser.write(data_packet.encode('utf-8'))
            cv2.putText(roi, f"Offset: {data_packet}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            time.sleep(0.02)
            if ser.in_waiting:
                recv_data = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"收到: {recv_data}")
            else:
                print("未收到下位机数据")
        else:
            # 红线丢失时，发送特殊六位数（比如 +99999）
            data_packet = "111111"
            ser.write(data_packet.encode('utf-8'))
            cv2.putText(roi, "111111", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            time.sleep(0.02)
            if ser.in_waiting:
                recv_data = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"收到: {recv_data}")
            else:
                print("未收到下位机数据")

        # 在主画面上画出ROI区域
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (0, roi_y1), (FRAME_WIDTH, FRAME_HEIGHT), (0,255,255), 2)
        # 把处理后的ROI贴回主画面
        debug_frame[roi_y1:, :] = roi

        cv2.imshow("Debug Line Follower", debug_frame)
        cv2.imshow("Red Mask", mask)

        key = cv2.waitKey(10)
        if key == 27:  # ESC退出
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()