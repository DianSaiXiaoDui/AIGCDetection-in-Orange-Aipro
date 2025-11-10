# -*- coding: utf-8 -*-
"""
SAFE_RINE Ascend 推理脚本（适配 utils 目录结构）
"""

import sys
import os
import cv2
import os
from collections import Counter
from pathlib import Path


# 确保当前目录加入模块搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from pytorch_wavelets import DWTForward
import acl

# 从 utils 目录导入 Ascend ACL 工具类（原 acllite）
import utils.acllite_utils as acl_utils
from utils.acllite_model import AclLiteModel
from utils.acllite_resource import AclLiteResource
from utils.smart_detection import saliency_based_crop, edge_density_crop, entropy_based_crop

# 获取当前脚本的绝对路径
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR # 根据实际情况调整层级


class SR_Ascend:
    def __init__(self, model_dir="./SR_om_models"):
        self.model_dir = model_dir

        # 初始化 ACL 资源（使用 utils 提供的 AclLiteResource）
        print("🚀 初始化 ACL 资源...")
        self.acl_resource = AclLiteResource()
        self.acl_resource.init()

        # 初始化 DWT 变换器（用于 SAFE 特征提取）
        #self.dwt = DWTForward(J=1, mode='symmetric', wave='bior1.3')

        # 图像预处理变换
        self.common_transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ])

        # CLIP 图像变换（224x224 + 归一化）
        '''
        self.clip_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        '''

        # 加载 OM 模型
        self._load_models()

    def _load_models(self):
        """加载三个 OM 模型：SAFE、CLIP、分类器"""
        print("🔧 加载 OM 模型...")
    
        # SAFE 特征提取模型
        safe_model_path = os.path.join(self.model_dir, "safe_feature.om")
        safe_size = os.path.getsize(safe_model_path) / 1024 / 1024
        self.safe_model = AclLiteModel(safe_model_path)
        #print(f"✅ 已加载 safe_feature.om, {safe_size:.2f} MB")

        # CLIP 特征提取模型
        clip_model_path = os.path.join(self.model_dir, "clip_feature_linux_aarch64.om")
        clip_size = os.path.getsize(clip_model_path) / 1024 / 1024
        self.clip_model = AclLiteModel(clip_model_path)
        #print(f"✅ 已加载 clip_feature_linux_aarch64.om, {clip_size:.2f} MB")

        # 分类器模型
        classifier_path = os.path.join(self.model_dir, "classifier.om")
        classifier_size = os.path.getsize(classifier_path) / 1024 / 1024
        self.classifier_model = AclLiteModel(classifier_path)
        #print(f"✅ 已加载 classifier.om, {classifier_size:.2f} MB")

        total_size = safe_size + clip_size + classifier_size
        print(f"✅ 模型加载成功，大小{total_size:.2f} MB")
    
    def _preprocess_dwt(self, x, mode='symmetric', wave='bior1.3'):
        '''
        pip install pywavelets pytorch_wavelets
        '''
        from pytorch_wavelets import DWTForward, DWTInverse
        DWT_filter = DWTForward(J=1, mode=mode, wave=wave).to(x.device)
        Yl, Yh = DWT_filter(x)
        return transforms.Resize([x.shape[-2], x.shape[-1]])(Yh[0][:, :, 2, :, :])

    def preprocess_image(self, image_path):
        """
        图像预处理：生成 SAFE 和 CLIP 两路输入
        """
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像不存在: {image_path}")
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path

        # 基础预处理：中心裁剪到 256x256
        image_tensor = self.common_transform(image).unsqueeze(0)  # [1, 3, 256, 256]
        #print(f"🖼️  原始图像预处理后: {image_tensor.shape}")
        '''
        crop_mode = "random"
        
        if crop_mode == "random":
            width, height = image.size
            i = torch.randint(0, height - 256 + 1, size=(1,)).item() if height > 256 else 0
            j = torch.randint(0, width - 256 + 1, size=(1,)).item() if width > 256 else 0
            
            cropped_image = image.crop((j, i, j + 256, i + 256))
            
        
        elif crop_mode == "saliency":
            cropped_image = saliency_based_crop(image)
            
        elif crop_mode == "edge_density":
            cropped_image = edge_density_crop(image)
            
        elif crop_mode == "entropy":
            cropped_image = entropy_based_crop(image)
            
        '''
        
        #cropped_image.save("region.png")
        
        # 基础预处理：使用裁剪后的图像继续处理
        #image_tensor = transforms.ToTensor()(cropped_image).unsqueeze(0)  # [1, 3, 256, 256]
            
        # SAFE 分支：DWT 提取 HH 频带
        safe_input_tensor = self._preprocess_dwt(image_tensor)  # [1, 3, 256, 256]
        safe_input = safe_input_tensor.squeeze(0).numpy().astype(np.float32)  # [3, 256, 256]

        # CLIP 分支：调整为 224x224 并归一化
        #clip_input_tensor = self.clip_transform(image_tensor)  # [1, 3, 224, 224]
        
        clip_input = image_tensor.squeeze(0).numpy().astype(np.float32) # [3, 244, 244]

        #print(f"✅ 预处理完成 - SAFE输入: {safe_input.shape}, CLIP输入: {clip_input.shape}")
        return safe_input, clip_input

    def extract_safe_feature(self, safe_input):
        """提取 SAFE 特征"""
        safe_input_batch = np.expand_dims(safe_input, axis=0)  # [1, 3, 256, 256]
        output = self.safe_model.execute([safe_input_batch])
        return output[0].flatten()  # 返回 512 维特征

    def extract_clip_feature(self, clip_input):
        """提取 CLIP 特征"""
        clip_input_batch = np.expand_dims(clip_input, axis=0)  # [1, 3, 224, 224]
        output = self.clip_model.execute([clip_input_batch])
        return output[0].flatten()  # 返回 1024 维特征

    def classify_features(self, safe_feature, clip_feature):
 
        # 推理分类器
        logits = self.classifier_model.execute([safe_feature, clip_feature])
        return logits[0].flatten()  # 返回 2维 logits

    def predict(self, image_path):
        """
        对单张图像进行预测
        """
        try:
            print(f"\n🖼️  处理图像: {image_path}")

            # 1. 预处理
            safe_input, clip_input = self.preprocess_image(image_path)

            # 2. 提取 SAFE 特征
            #print("🔍 提取 SAFE 特征...")
            safe_feature = self.extract_safe_feature(safe_input)
            #print(f"✅ SAFE 特征维度: {safe_feature.shape}")

            # 3. 提取 CLIP 特征
            #print("🔍 提取 CLIP 特征...")
            clip_feature = self.extract_clip_feature(clip_input)
            #print(f"✅ CLIP 特征维度: {clip_feature.shape}")

            # 4. 分类
            #print("🔍 分类推理...")
            logits = self.classify_features(safe_feature, clip_feature)
            print(f"✅ 分类 logits: {logits}")

            # 5. 后处理：Softmax 得到概率
            logits_tensor = torch.tensor(logits).unsqueeze(0)  # [1, 2]
            probs = torch.softmax(logits_tensor, dim=1)[0]
            fake_prob = probs[1].item()
            true_prob = probs[0].item()
            #pred_class = 1 if fake_prob > real_prob else 0

            adaptive_thres = True
            # 5.计算置信度
            if adaptive_thres:
                thres = 1e-21
                pred_class = 1 if fake_prob > thres else 0
                if pred_class == 0:
                   confidence = (0.5 / thres) * (true_prob - 1 + thres) + 0.5
                else:
                   confidence = 0.5 / (1-thres) * (fake_prob - thres)  + 0.5  
                #print(f"logits_tensor:{logits_tensor}\nprobs:{probs}")
            else:
                pred_class = 1 if fake_prob > 0.5 else 0
                confidence = fake_prob if pred_class == 1 else true_prob
            result = {
                'prediction': 'Fake' if pred_class == 1 else 'True',
                'confidence': confidence
            }

            print(f"🎉 预测结果: {result}")
            return result

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'prediction': 'error',
                'confidence': 0.0,
                'error': str(e)
            }


def check_environment():
    """检查运行环境是否完整"""
    print("🧪 检查运行环境...")

    # 检查模型目录
    model_dir = PROJECT_ROOT / "./SR_om_models"
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        return False

    required_models = [
        "safe_feature.om",
        "clip_feature_linux_aarch64.om",
        "classifier.om"
    ]
    missing = []
    size = 0
    for model in required_models:
        path = os.path.join(model_dir, model)
        if not os.path.exists(path):
            missing.append(model)
 
    if missing:
        print(f"❌ 缺少模型文件: {missing}")
        return False

    return True

def SR_Init():
    detector = SR_Ascend(model_dir = PROJECT_ROOT / "SR_om_models")
    return detector
    
def SR_DeInit(detector = None):
    del detector
    print("🧹 资源清理完成")
       

# 项目接口一:SAFE_RINE对输入图像进行推理, 输出图像真假类别和置信
def ISID(detector, test_img):
    print("=" * 60)
    print("🚀 SAFE_RINE Ascend 推理引擎启动")
    print("=" * 60)

    # 环境检查
    if not check_environment():
        print("❌ 环境检查失败，请检查模型文件")
        sys.exit(1)

    try:
        print("\n🔧 正在初始化模型...")
        print(f"绝对路径:{os.path.abspath(test_img)}")

        # 测试图像路径
        #test_img = "test_data/0_real/0.jpg"
        if os.path.exists(test_img):
            print(f"📸 开始推理测试图像: {test_img}")
            result = detector.predict(test_img)

            print(f"\n📊 最终预测结果:")
            print(f"   类别: {result['prediction']}")
            print(f"   置信度: {result['confidence']:.4f}")
            return result['prediction'], result['confidence']
        else:
            print(f"⚠️  测试图像未找到: {test_img}")
            print("💡 请将测试图像放入 test_data/ 目录或指定路径")
        

    except Exception as e:
        print(f"❌ 程序异常终止: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    #finally:
    #    if detector:
    #        del detector
    #    print("🧹 资源清理完成")


def extract_frames(video_path, num_frames=8):
    """
    从视频中均匀提取 num_frames 帧
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    # 均匀采样时间点
    if total_frames <= num_frames:
        frame_indices = range(total_frames)
    else:
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            # 转为 RGB 并转换为 PIL Image 格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)

    cap.release()
    print(f"✅ 从视频中提取 {len(frames)} 帧用于检测")
    return frames


def extract_frames(video_path, num_frames=8):
    """
    从视频中均匀提取 num_frames 帧
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    # 均匀采样时间点
    if total_frames <= num_frames:
        frame_indices = range(total_frames)
    else:
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            # 转为 RGB 并转换为 PIL Image 格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append(pil_image)

    cap.release()
    print(f"✅ 从视频中提取 {len(frames)} 帧用于检测")
    return frames

# 项目接口二:SAFE_RINE对输入视频抽取若干帧进行推理, 综合几帧推理结果，返回视频检测真假结果和平均置信度
def VSID(video_path, num_frames=8, threshold=0.5):
    """
    伪造视频检测主接口
    Args:
        video_path (str): 视频文件路径
        num_frames (int): 采样帧数，默认 8
        threshold (float): 判定为 fake 的置信度阈值（用于单帧判断，可选）

    Returns:
        dict: {'prediction': 'fake/real', 'confidence': float}
    """
    print("=" * 60)
    print("🎥 SAFE_RINE 视频伪造检测启动")
    print("=" * 60)

    try:
        # 1. 提取帧
        frames = extract_frames(video_path, num_frames=num_frames)
        if len(frames) == 0:
            print("❌ 未提取到任何有效帧")
            return {'prediction': 'error', 'confidence': 0.0, 'error': 'no frames extracted'}

        # 2. 初始化检测器（复用 SID 中的模型）
        print("\n🔧 初始化 SAFE_RINE 模型...")
        detector = SR_Ascend(model_dir="./SR_om_models")

        predictions = []
        confidences = []

        # 3. 遍历每一帧进行检测
        for idx, frame in enumerate(frames):
            print(f"\n🖼️  处理第 {idx+1}/{len(frames)} 帧...")
            result = detector.predict(frame)  # 注意：这里传入的是 PIL.Image

            pred = result['prediction']
            conf = result['confidence']

            predictions.append(pred)
            confidences.append(conf if pred == 'fake' else -conf)  # fake 用正数，real 用负数便于平均

            print(f"   帧 {idx+1} 预测: {pred}, 置信度: {result['confidence']:.4f}")

        # 4. 综合判断
        fake_count = sum(1 for p in predictions if p == 'fake')
        real_count = len(predictions) - fake_count

        # 多数投票
        final_pred = 'fake' if fake_count > real_count else 'real'

        # 平均“加权置信度”：fake 为正，real 为负，取绝对值后加权平均
        avg_confidence = np.mean([abs(c) for c in confidences])
        
        # 更精细：按投票比例加权
        vote_ratio = fake_count / len(predictions)
        calibrated_conf = avg_confidence * (2 * abs(vote_ratio - 0.5))  # 强化多数票的置信

        final_result = {
            'prediction': final_pred,
            'confidence': float(calibrated_conf),
            'frame_count': len(predictions),
            'fake_frame_count': fake_count,
            'real_frame_count': real_count,
            'per_frame_predictions': predictions,
            'per_frame_confidences': [float(c) for c in confidences]
        }

        print(f"\n📊 视频最终预测结果:")
        print(f"   类别: {final_result['prediction']}")
        print(f"   置信度: {final_result['confidence']:.4f}")
        print(f"   详细: {fake_count} 帧判为 fake, {real_count} 帧判为 real")

        return final_result

    except Exception as e:
        print(f"❌ 视频检测失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'prediction': 'error',
            'confidence': 0.0,
            'error': str(e)
        }

    finally:
        # 清理资源
        if 'detector' in locals():
            del detector
        print("🧹 视频检测资源清理完成")

def main():

    #ISID example 1
    test_img = "test_images/1_fake/example_fake.jpeg"
    
    detector = SR_Init()
    label, confidence = ISID(detector, test_img)
    SR_DeInit(detector)
    #VSID example 2
    #test_video = "test_videos/fake_2.mp4"  
    #final_result = VSID(test_video)  

if __name__ == "__main__":
    main()