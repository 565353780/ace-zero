import os
import cv2
from tqdm import tqdm


def videoToImages(
    video_file_path: str,
    save_image_folder_path: str,
    down_sample_scale: int=1,
    scale: float=1,
    show_image: bool=False,
    print_progress: bool=False,
) -> bool:
    if not os.path.exists(video_file_path):
        print('[ERROR][video::videoToImages]')
        print("\t video file not exist!")
        print('\t video_file_path:', video_file_path)
        return False

    if save_image_folder_path[-1] != "/":
        save_image_folder_path += "/"

    os.makedirs(save_image_folder_path, exist_ok=True)

    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        print('[ERROR][video::videoToImages]')
        print("\t video file can not open!")
        print('\t video_file_path:', video_file_path)
        return False

    total_image_num = int(cap.get(7))

    for_data = range(total_image_num)
    if print_progress:
        print("[INFO][video::videoToImages]")
        print("\t start convert video to images...")
        for_data = tqdm(for_data)
    for image_idx in for_data:
        status, frame = cap.read()
        if not status:
            break

        image_idx += 1

        if image_idx % down_sample_scale != 0:
            continue

        if scale != 1:
            frame = cv2.resize(
                frame, (int(frame.shape[1] / scale), int(frame.shape[0] / scale))
            )

        if show_image:
            cv2.imshow("image", frame)
            cv2.waitKey(1)

        save_image_file_path = (
            save_image_folder_path + "image_" + str(image_idx) + ".png"
        )
        cv2.imwrite(save_image_file_path, frame)

    cap.release()
    if show_image:
        cv2.destroyAllWindows()
    return True
