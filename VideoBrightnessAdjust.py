import os
import cv2
from PIL import Image, ImageEnhance

if __name__ == '__main__':
    input_video = cv2.VideoCapture('C:\\学习资料\\项目\\videotarget\\yeyou.mp4')
    input_fps = input_video.get(cv2.CAP_PROP_FPS)
    success, image = input_video.read()
    count = 0 #想要截取的视频起始帧

    print("Converting Video Into Frames and Improvising their brightness")
    while success: #想要截取的视频结束帧
        cv2.imwrite("frame%d.jpg" % count, image)
        success, image = input_video.read()

        frame = Image.open("frame%d.jpg" % count)
        enhancer = ImageEnhance.Brightness(frame)
        factor = 1.5
        im_output = enhancer.enhance(factor)
        im_output.save("processed_frame%d.jpg" % count)

        os.remove("frame%d.jpg" % count)
        count += 1

    # Generating Video from frames
    print("Generating Video from Improvised Frames")
    video_name = 'output_video.mp4'
    frame = cv2.imread("processed_frame%d.jpg" % 0)
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), input_fps, (width, height))

    # Appending the images to the video one by one
    for i in range(count):
        video.write(cv2.imread("processed_frame%d.jpg" % i))
        os.remove("processed_frame%d.jpg" % i)

    cv2.destroyAllWindows()
    video.release()
    print("Video Improvisation Completed")