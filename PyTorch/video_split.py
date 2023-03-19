import cv2  # 导入opencv模块
import os
import time


def video_split(video_path, save_path):
    '''
    对视频文件切割成帧
    '''
    '''
    @param video_path:视频路径
    @param save_path:保存切分后帧的路径
    '''
    vc = cv2.VideoCapture(video_path)
    # 一帧一帧的分割 需要几帧写几
    c = 0
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    while rval:
        rval, frame = vc.read()
        # 每秒提取5帧图片
        if c % 5 == 0:
            cv2.imwrite(save_path + "/" + str('%06d' % c) + '.jpg', frame)
            cv2.waitKey(1)
        c = c + 1


DATA_DIR = "UCF-101_"  # 视频数据主目录

SAVE_DIR = "UCF-101"  # 帧文件保存目录

start_time = time.time()
for parents, dirs, filenames in os.walk(DATA_DIR):
    if parents == DATA_DIR:
        continue

    print("正在处理文件夹", parents)
    path = parents.replace("\\", "//")
    f = parents.split("\\")[1]
    save_path = SAVE_DIR + "//" + f
    # 对每视频数据进行遍历
    for file in filenames:
        file_name = file.split(".")[0]
        save_path_ = save_path + "/" + file_name
        if not os.path.isdir(save_path_):
            os.makedirs(save_path_)
        video_path = path + "/" + file
        video_split(video_path, save_path_)

end_time = time.time()
print("Cost time", start_time - end_time)