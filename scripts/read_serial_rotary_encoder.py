#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
串口角度读取器 - 带TUI界面
功能：
- 以指定波特率打开串口
- 以10Hz频率发送HEX数据
- 接收并解析角度数据
- 显示带状态的TUI界面
"""

import serial
import threading
import time
import sys
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich import box
import keyboard
import rerun as rr
import pyfiglet


class AngleReader:
    def __init__(self, port="COM4", baudrate=1000000):
        """
        初始化角度读取器
        
        Args:
            port (str): 串口号，默认COM4
            baudrate (int): 波特率，默认1M
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_running = False
        self.console = Console()
        
        # 通信数据
        self.send_data = bytes.fromhex("01 03 00 41 00 01 d4 1e")
        self.last_received = b""
        self.last_angle = 0.0
        self.send_count = 0
        self.receive_count = 0
        self.error_count = 0
        
        # 状态信息
        self.status = {
            'port': port,
            'baudrate': baudrate,
            'connected': False,
            'last_send': '',
            'last_receive': '',
            'angle': 0.0,
            'angle_raw': 0,
            'send_count': 0,
            'receive_count': 0,
            'error_count': 0
        }

        # rr.init("GripperWidthCorrection", spawn=True)
    
    def connect_serial(self):
        """连接串口"""
        # try:
        self.serial_conn = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1
        )
        self.status['connected'] = True
        return True
        # except Exception as e:
        #     self.console.print(f"[red]串口连接失败: {e}[/red]")
        #     return False
    
    def disconnect_serial(self):
        """断开串口连接"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.status['connected'] = False
    
    def calculate_angle(self, data_bytes):
        """
        计算角度值
        
        Args:
            data_bytes (bytes): 接收到的数据
            
        Returns:
            float: 计算出的角度值
        """
        try:
            if len(data_bytes) >= 7:
                # 提取角度数据 (第4和第5字节)
                angle_high = data_bytes[3]
                angle_low = data_bytes[4]
                angle_raw = (angle_high << 8) | angle_low
                mask = 0b111111111111
                data_bit = 0b100000000000
                angle_raw = angle_raw & mask

                if (angle_raw & data_bit):
                    angle_raw = -((~angle_raw & mask) + 1)

                self.status['angle_raw'] = angle_raw

                # print(f"原始角度值: {angle_raw}")
                
                # 计算角度: 360 * raw_value / 1024
                angle = 360.0 * angle_raw / 4096.0
                return angle
        except Exception as e:
            self.error_count += 1
            return 0.0
        
        return 0.0
    
    def send_command(self):
        """发送命令到串口"""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(self.send_data)
                self.send_count += 1
                self.status['send_count'] = self.send_count
                self.status['last_send'] = ' '.join([f"{b:02X}" for b in self.send_data])
                return True
            except Exception as e:
                self.error_count += 1
                return False
        return False
    
    def read_response(self):
        """读取串口响应"""
        if self.serial_conn and self.serial_conn.is_open:
            try:
                # 尝试读取数据
                data = self.serial_conn.read(10)  # 读取最多10字节
                if data:
                    self.last_received = data
                    self.receive_count += 1
                    self.status['receive_count'] = self.receive_count
                    self.status['last_receive'] = ' '.join([f"{b:02X}" for b in data])
                    
                    # 计算角度
                    angle = self.calculate_angle(data)
                    self.last_angle = angle
                    self.status['angle'] = angle
                    rr.log("Angle", rr.Scalars(angle))

                    return data
            except Exception as e:
                self.error_count += 1
                return None
        return None
    
    def communication_loop(self):
        """通信循环 - 在单独线程中运行"""
        while self.is_running:
            # 发送命令
            if self.send_command():
                # 等待短暂时间后读取响应
                # time.sleep(0.01)  # 10ms延迟
                self.read_response()
            
            # 更新状态
            self.status['error_count'] = self.error_count
            
            # 10Hz频率 (100ms间隔)
            # time.sleep(0.01)
    
    def create_layout(self):
        """创建TUI布局"""
        layout = Layout()
        
        # 主要分为三部分：标题、角度显示、状态信息
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=2),
            Layout(name="status", size=8)
        )
        
        # 状态区域分为两列
        layout["status"].split_row(
            Layout(name="left_status"),
            Layout(name="right_status")
        )
        
        return layout
    
    def update_display(self, layout):
        """更新显示内容"""
        # 标题
        layout["header"].update(
            Panel(
                Text("串口角度读取器 - 按 'q' / Ctrl+C 退出", style="bold white", justify="center"),
                style="blue",
                box=box.ROUNDED
            )
        )
        
        # 主要角度显示
        angle_text = Text(f"{self.status['angle']:.3f}°", style="bold green")
        angle_text.stylize("bold", 0, len(angle_text))
        
        layout["main"].update(
            Panel(
                Text(f"{self.status['angle']:.3f}°", style="bold green", justify="center"),
                title="当前角度",
                style="green",
                box=box.DOUBLE
            )
        )
        
        # 左侧状态信息
        left_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        left_table.add_column("项目", style="cyan")
        left_table.add_column("值", style="white")
        
        connection_status = "✓ 已连接" if self.status['connected'] else "✗ 未连接"
        connection_style = "green" if self.status['connected'] else "red"
        
        left_table.add_row("串口:", f"{self.status['port']}")
        left_table.add_row("波特率:", f"{self.status['baudrate']}")
        left_table.add_row("状态:", Text(connection_status, style=connection_style))
        left_table.add_row("发送次数:", f"{self.status['send_count']}")
        left_table.add_row("接收次数:", f"{self.status['receive_count']}")
        left_table.add_row("错误次数:", f"{self.status['error_count']}")
        
        layout["left_status"].update(
            Panel(
                left_table,
                title="连接状态",
                style="blue"
            )
        )
        
        # 右侧通信数据
        right_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        right_table.add_column("项目", style="cyan")
        right_table.add_column("数据", style="yellow")
        
        right_table.add_row("发送:", f"{self.status['last_send']}")
        right_table.add_row("接收:", f"{self.status['last_receive']}")
        right_table.add_row("原始值:", f"{self.status['angle_raw']}")
        
        layout["right_status"].update(
            Panel(
                right_table,
                title="通信数据",
                style="blue"
            )
        )
    
    def run(self):
        """运行主程序"""
        self.console.print("[yellow]正在启动串口角度读取器...[/yellow]")

        # 连接串口
        if not self.connect_serial():
            self.console.print("[red]无法连接串口，程序退出[/red]")
            return
        
        self.console.print(f"[green]串口 {self.port} 连接成功，波特率: {self.baudrate}[/green]")
        
        # 创建布局
        layout = self.create_layout()
        
        # 启动通信线程
        self.is_running = True
        comm_thread = threading.Thread(target=self.communication_loop, daemon=True)
        comm_thread.start()
        
        try:
            # 使用Rich Live显示
            with Live(layout, refresh_per_second=100, screen=True) as live:
                while self.is_running:
                    self.update_display(layout)
                    live.update(layout)
                    
                    # 检查退出按键
                    try:
                        if keyboard.is_pressed('q'):
                            break
                    except ImportError:
                        continue
                    
                    # time.sleep(0.01)
        
        except KeyboardInterrupt:
            pass
        finally:
            # 清理资源
            self.is_running = False
            self.disconnect_serial()
            self.console.print("\n[yellow]程序已退出[/yellow]")


def main():
    """主函数"""
    # 可以在这里修改串口参数
    # PORT = "COM4"        # 串口号 on Windows
    PORT = "/dev/ttyUSB0"
    BAUDRATE = 1000000   # 波特率 (1M)
    
    # try:
    reader = AngleReader(port=PORT, baudrate=BAUDRATE)
    reader.run()
    # except Exception as e:
    #     print(f"程序运行出错: {e}")
    #     sys.exit(1)


if __name__ == "__main__":
    main()