import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor

def process_single_video(input_path, output_path):
    # 检查视频是否有音频
    command_check_audio = [
        'ffmpeg', '-i', input_path, '-af', 'volumedetect', '-f', 'null', 'NUL'
    ]
    result = subprocess.run(command_check_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if 'mean_volume' not in result.stderr:
        return False

    # 调整大小和裁剪视频
    command_process_video = [
        'ffmpeg', '-i', input_path,  # 输入文件
        '-vf', 'scale=224:224',  # 调整视频大小
        '-r', '20',  # 设置帧率
        '-c:v', 'libx264',  # 使用 H.264 编码器
        '-preset', 'fast',  # 编码速度
        '-crf', '23',  # 压缩质量
        output_path  # 输出文件
    ]
    subprocess.run(command_process_video, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return True

def process_video(input_dir, output_dir, skip_count):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tasks = []
    total_count = 0
    processed_count = 0
    skipped_count = 0

    for root, dirs, files in os.walk(input_dir):
        for dir in dirs:
            os.makedirs(os.path.join(output_dir, dir), exist_ok=True)
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                total_count += 1

    with ThreadPoolExecutor(max_workers=16) as executor:  # 调整 max_workers 以适应你的计算机性能
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(('.mp4', '.avi')):
                    if skipped_count < skip_count:
                        skipped_count += 1
                        if(skipped_count % 50 == 0):
                            print(f"Skipped {skipped_count} videos")
                        continue
                    input_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, input_dir)
                    output_path = os.path.join(output_dir, relative_path, file)
                    tasks.append(executor.submit(process_single_video, input_path, output_path))

        for task in tasks:
            if task.result():
                processed_count += 1
                if processed_count % 50 == 0 or processed_count == total_count - skip_count:
                    print(f"Processed {processed_count}/{total_count - skip_count} videos")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos")
    parser.add_argument('--input_dir', type=str, required=True, help="Input directory containing videos")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for processed videos")
    parser.add_argument('--skip_count', type=int, default=0, help="Number of videos to skip")
    args = parser.parse_args()

    process_video(args.input_dir, args.output_dir, args.skip_count)