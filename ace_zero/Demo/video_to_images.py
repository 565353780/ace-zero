from ace_zero.Method.video import videoToImages

def demo():
    video_file_path = '/home/lichanghao/chLi/Dataset/GS/haizei_1.MOV'
    save_image_folder_path = '/home/lichanghao/chLi/Dataset/GS/haizei_1/images/'
    down_sample_scale = 1
    scale = 1
    show_image = False
    print_progress = True

    videoToImages(
        video_file_path,
        save_image_folder_path,
        down_sample_scale,
        scale,
        show_image,
        print_progress,
    )
    return True
