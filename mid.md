# 中期报告

## 选题：结合视频和音频信息完成分类任务

### 实验目标与背景

本实验的内容是实现一个结合视觉和音频信息的视频动作分类模型。我们将采用**Vision Transformer (ViT)** 作为基础模型，并通过集成**LAVISH**适配器来实现视觉和音频的融合任务，探索如何在不改变原ViT参数的情况下利用音频信息提升分类性能。

实验的主要目标复现论文《Vision transformers are parameter-efficient audio-visual learners》（Lin, Yan-Bo, et al., 2023），该论文通过LAVISH适配器结合视觉和音频数据，提升了ViT在音视频融合任务中的表现。

论文中主要评估了以下三类任务：

- **AVE**（Audio-Visual Event Localization）
- **AVQE**（Audio-Visual Question Answering）
- **AVS**（Audio-Visual Segmentation）

我们的重点是将**LAVISH**模块应用到视频动作分类任务上，使用UCF101数据集进行实验。

## 进展情况

### 数据集准备

我们使用了**UCF-101**数据集，这是一个包含13320个视频样本和101个动作类别的数据集，广泛用于视频分类任务。该数据集提供了各种视频样本，涵盖了多种动作类别，如体育运动、日常活动等。为了确保只处理带有音频的数据，我们从UCF-101数据集中筛选出了包含音频的样本进行处理。

### 预处理与数据增强

我们参考了https://github.com/Alexyuda/action-recognition-video-audio中的数据预处理流程。具体的预处理步骤如下：

1. **视频格式转换**：将视频的尺寸统一调整为224x224，并将每个视频裁剪为20帧。
2. **音频数据处理**：由于本项目关注的是多模态信息，我们仅选择包含音频的数据，并将其提取和处理为可用的特征。
3. **数据划分**：根据默认的训练集和测试集划分，UCF-101数据集中的8548个视频被用于训练，3260个视频用于测试。

最后，为了适应模型的训练需求，我们将每个视频划分为多个clip，每64帧一个clip，并且采用每隔64帧取一个clip的方式进行数据预处理，确保数据的多样性和丰富性。

```python
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
    result = subprocess.run(command_process_video, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"FFmpeg failed for {input_path}. Error:\n{result.stderr}")
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

    with ThreadPoolExecutor(max_workers=16) as executor:  # 调整 max_workers 以适应计算机性能
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
```



### Baseline模型准备

我们参考了**action-recognition-video-audio**中提供的Baseline模型，分别使用了以下几种组合方式进行测试：

- **i3d模型**（纯视觉）
- **soundnet + i3d模型**（音频和视觉的融合）
- **i3d + soundnet + attention机制模型**（更加复杂的多模态融合）

这些方法都表现得非常优秀，在测试集上Top1准确率超过0.9，Top5准确率超过0.98。

### ViT模型实验

接下来，我们准备了**ViT模型**作为基线模型，分为以下两种情景进行实验：

1. **单模态视觉模型**：使用视觉信息单独进行训练，预训练模型选用`google/vit-base-patch16-224`。
2. **多模态融合模型**：将视觉和音频信息进行融合，使用Late Fusion方法结合视觉和音频特征进行联合训练。

但是在实验过程中我们遇到了一些挑战，尤其是在训练速度上。由于使用的**ViT预训练模型**具有较大的计算开销，并且每个视频clip的尺寸为3x64x224x224，导致显存消耗非常大。在使用**batch_size = 1**的情况下，单个epoch需要大约3天时间才能完成一次训练。

为了解决这个问题，我们考虑通过**减少clip长度**来增大batch size，从而提高训练效率并减少显存占用。

以下是将clip减小到1帧的训练结果：

![ff12b17af465a576431fb14bfa4900c](E:\HuaweiMoveData\Users\胡梦溪\Documents\WeChat Files\wxid_b34vgato91qt22\FileStorage\Temp\ff12b17af465a576431fb14bfa4900c.png)

### 持续优化与实验挑战

当前，我们的实验仍面临以下挑战：

- **训练速度**：由于模型复杂，训练速度较慢，需要优化训练管道或通过减少输入尺寸来加速训练。
- **资源限制**：由于显存限制，当前的batch size设置较小，且计算消耗较高。需要在保证精度的前提下找到合适的平衡点。
- **模型优化**：虽然初步的实验表现良好，但仍需要对ViT和LAVISH模块进行优化，尤其是多模态融合部分，确保其在不同任务上的适应性。

## 后续工作规划

### 下一步的任务

1. **模型优化与调试**：
   - 对ViT模型和LAVISH模块进行进一步的调优，提升训练效率。
   - 继续优化音频和视觉的多模态融合方法，尝试加上LAVISH的module实现一个版本，提升融合效果。
2. **扩展实验**：
   - 深入分析Bad Case，识别模型在特定情形下表现不佳的原因，并进行改进。
3. **可视化与分析**：
   - 利用**TSNE**和**PCA**对模型的特征空间进行进一步的可视化，探索模型在不同特征空间的表现。
   - 分析模型在不同任务中的表现，是否更依赖于视觉特征还是音频特征。

## 总结

到目前为止，项目已经完成了数据集的预处理和基线模型的训练。后续将尝试实现LAVISH模块的集成，继续优化模型、扩展实验范围，并深入分析模型在多模态融合中的优势与不足。

