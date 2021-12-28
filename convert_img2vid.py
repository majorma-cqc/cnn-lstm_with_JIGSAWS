import cv2
img = cv2.imread('./grad_cam_pics/1_frame/grad_1.jpg')
imgInfo = img.shape
size = (imgInfo[1], imgInfo[0])

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
videoWrite = cv2.VideoWriter( './grad_cam_pics/vdieo_get/2.avi', fourcc, 5, size )
# 写入对象 1 file name 2 编码器 3 帧率 4 尺寸大小

for i in range(1, 310):
    fileName = './grad_cam_pics/1_frame/grad_'+str(i)+'.jpg'
    img = cv2.imread(fileName)
    imgInfo = img.shape
    videoWrite.write(img) # 写入方法 1 jpg data

print('end1')
pass