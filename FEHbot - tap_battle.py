import time
import win32api
import win32con
import win32gui
import win32ui
import cv2
import pygetwindow
import numpy as np

# Bluestacks has a children window to send mouse click events to so we get that


def getchildwin(winame):
    hwnd = win32gui.FindWindow(None, str(winame))
    child_handles = []
    def all_ok(hwnd, param): child_handles.append(hwnd)
    win32gui.EnumChildWindows(hwnd, all_ok, None)
    return child_handles


handle = getchildwin("BlueStacks")[0]

# We then define a method to click inside the window using an event click


def clickwin(x, y):
    lParam = win32api.MAKELONG(x, y)
    win32gui.SendMessage(handle, win32con.WM_LBUTTONDOWN,
                         win32con.MK_LBUTTON, lParam)
    win32gui.SendMessage(handle, win32con.WM_LBUTTONUP, None, lParam)
    time.sleep(0.1)

# This is used to keep track of what's on the bluestacks window


def get_screenshot():
    # get the window image data
    hwnd = win32gui.FindWindow(None, "BlueStacks")
    x, y, w, h = win32gui.GetWindowRect(hwnd)
    width = w-x
    height = h-y
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, width, height)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (width, height), dcObj, (0, 0), win32con.SRCCOPY)
    # convert the raw data into a format opencv can read
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)
    # free resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    img = img[..., :3]
    img = np.ascontiguousarray(img)

    return img

# Use openCV2 to find if the required button is on screen at the time and click it if it is


def isonscreen(small_image, large_image):
    method = cv2.TM_SQDIFF_NORMED
    res = cv2.matchTemplate(small_image, large_image, cv2.TM_CCOEFF_NORMED)
    min_v, max_v, min_pt, max_pt = cv2.minMaxLoc(res)
    if max_v > 0.85:
        clickwin(max_pt[0], max_pt[1])
        return True
    return False

def isonscreen_noclick(small_image, large_image):
    method = cv2.TM_SQDIFF_NORMED
    res = cv2.matchTemplate(small_image, large_image, cv2.TM_CCOEFF_NORMED)
    min_v, max_v, min_pt, max_pt = cv2.minMaxLoc(res)
    if max_v > 0.85:
        return True
    return False

def getposition(small_image, large_image):
    method = cv2.TM_SQDIFF_NORMED
    res = cv2.matchTemplate(small_image, large_image, cv2.TM_CCOEFF_NORMED)
    min_v, max_v, min_pt, max_pt = cv2.minMaxLoc(res)
    if max_v > 0.85:
        return (max_pt[0], min_pt[1])
    return False

# Resize windows to the correct size
win = pygetwindow.getWindowsWithTitle('BlueStacks')[0]
win.size = (499, 862)

# Import all the buttons screenshots
enemies = cv2.imread("tapbattle/tapbattle_1.png")
button = cv2.imread("tapbattle/tapbattle_2.png")
hearts = cv2.imread("tapbattle/hearts.png")
corner = cv2.imread("tapbattle/corner.png")
skip = cv2.imread("tapbattle/skip.png")
stageclear = cv2.imread("tapbattle/stageclear.png")
close  = cv2.imread("tapbattle/close.png")
ok = cv2.imread("tapbattle/ok.png")

counter = 0
time_start = time.time()

# Run the bot
while 1:
    screenshot = get_screenshot()
    if not isonscreen_noclick(enemies, screenshot) and isonscreen_noclick(hearts, screenshot):
        clickwin(125, 732)
    isonscreen(skip, screenshot)
    isonscreen(stageclear, screenshot)
    isonscreen(close, screenshot)
    isonscreen(ok, screenshot)