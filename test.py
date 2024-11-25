import os
import subprocess
import argparse

def check_audio_in_video(input_path):
    # 检查视频是否有音频
    command_check_audio = [
        'ffmpeg', '-i', input_path, '-af', 'volumedetect', '-f', 'null', 'NUL'
    ]
    result = subprocess.run(command_check_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return 'mean_volume' in result.stderr

def count_videos_with_audio(input_dir):
    total_videos = 0
    videos_with_audio = 0

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                total_videos += 1
                input_path = os.path.join(root, file)
                if check_audio_in_video(input_path):
                    videos_with_audio += 1
            if total_videos % 50 == 0:
                print(f"Processed {total_videos} videos")
    return total_videos, videos_with_audio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count videos with audio")
    parser.add_argument('--input_dir', type=str, required=True, help="Input directory containing videos")
    args = parser.parse_args()

    total_videos, videos_with_audio = count_videos_with_audio(args.input_dir)
    print(f"Total videos: {total_videos}")
    print(f"Videos with audio: {videos_with_audio}")