from mylib import *
import win32api
import win32gui
import win32con

TOOL_GROUP_TEXT = '快捷工具栏'
CONFIG_BUTTON_TEXT = '配置芯片'

print(get_process_count('EZPro100.exe'))

hwnd_main = win32gui.FindWindow(None, 'EZPro100')
hwnd_group_tool = win32gui.FindWindowEx(hwnd_main, 0, None, TOOL_GROUP_TEXT)
hwnd_cmd_config = win32gui.FindWindowEx(
    hwnd_group_tool, 0, None, CONFIG_BUTTON_TEXT)
win32api.PostMessage(hwnd_cmd_config, win32con.BM_CLICK, 0, 0)
