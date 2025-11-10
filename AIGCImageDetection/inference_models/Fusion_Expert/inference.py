# -*- coding: utf-8 -*-
"""
Fusion_Expert_RINE Ascend 推理脚本（适配 utils 目录结构）
"""

import sys
import os
import cv2
import os
from pathlib import Path
from collections import Counter


# 确保当前目录加入模块搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from pytorch_wavelets import DWTForward
import acl

# 从 utils 目录导入 Ascend ACL 工具类
import acl
import utils.acllite_utils as acl_utils
from utils.acllite_model import AclLiteModel
from utils.acllite_resource import AclLiteResource
from utils.smart_detection import saliency_based_crop, edge_density_crop, entropy_based_crop

# 获取当前脚本的绝对路径
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR # 根据实际情况调整层级

# 设置项目根目录为工作目录
#os.chdir(PROJECT_ROOT)


class Fusion_Expert_Ascend:
    def __init__(self, model_dir= "./om_models"):
        self.model_dir = PROJECT_ROOT / model_dir

        # 初始化 ACL 资源（使用 utils 提供的 AclLiteResource）
        print("🚀 初始化 ACL 资源...")
        self.acl_resource = AclLiteResource()
        self.acl_resource.init()


        # 图像预处理变换
        self.common_transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
               std = [0.229, 0.224, 0.225])  
        ])

        # 加载 OM 模型
        self._load_models()

    def _load_models(self):
        """加载 OM 模型：Fusion_Expert"""
        print("🔧 加载 OM 模型...")
    
        # Fusion_Expert 模型
        model_path = os.path.join(self.model_dir, "fusion_expert.om")
        model_size = os.path.getsize(model_path) / 1024 / 1024
        self.model = AclLiteModel(model_path)
        
        print(f"✅ 已加载 fusion_expert.om, {model_size:.2f} MB")


    def preprocess_image(self, image_path):
        """
        图像预处理：中心裁剪224+ImageNet权重标准化
        """
        # 确保处理所有可能的输入类型
        if isinstance(image_path, (str, Path)):
            # 如果是字符串或Path对象，打开图像
            if not os.path.exists(str(image_path)):
                raise FileNotFoundError(f"图像不存在: {image_path}")
            image = Image.open(str(image_path)).convert('RGB')
        elif isinstance(image_path, Image.Image):
            # 如果已经是PIL Image，直接使用
            image = image_path
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            raise ValueError(f"不支持的图像输入类型: {type(image_path)}")
        
        crop_mode = "saliency"
        print(f"crop mode:{crop_mode}")
        
        if crop_mode == "random":
            width, height = image.size
            i = torch.randint(0, height - 256 + 1, size=(1,)).item() if height > 256 else 0
            j = torch.randint(0, width - 256 + 1, size=(1,)).item() if width > 256 else 0
            
            cropped_image = image.crop((j, i, j + 256, i + 256))
            
        
        elif crop_mode == "saliency":
            cropped_image = saliency_based_crop(image, 224)
            
        elif crop_mode == "edge_density":
            cropped_image = edge_density_crop(image)
            
        elif crop_mode == "entropy":
            cropped_image = entropy_based_crop(image)
            
        elif crop_mode == None:    
            image_tensor = self.common_transform(image).unsqueeze(0)  # [1, 3, 256, 256]
        #cropped_image.save("region.png")
        #print(f"✅ 裁剪区域已保存: region.png")    

        # 基础预处理：使用裁剪后的图像继续处理
        if crop_mode:
           image_tensor = transforms.ToTensor()(cropped_image) # [1, 3, 256, 256]
           image_tensor =  transforms.Normalize(mean = [0.485, 0.456, 0.406],
               std = [0.229, 0.224, 0.225]) (image_tensor) .unsqueeze(0)


        #print(f"🖼️  原始图像预处理后: {image_tensor.shape}")
        # 先执行随机裁剪并保存
        model_input = image_tensor.squeeze(0).numpy().astype(np.float32)  # [3, 256, 256]


        return model_input

    def body_inference(self, model_input):
 
        # 推理分类器
        model_input_batch = np.expand_dims(model_input, axis=0)  # [1, 3, 256, 256]
        logits = self.model.execute([model_input_batch])
        #print(f"logits：{logits}")
        return logits[0].flatten()  # 返回 2维 logits

    def predict(self, image_path):
        """
        对单张图像进行预测
        """
        try:
            print(f"\n🖼️  处理图像: {image_path}")

            # 1. 预处理
            model_input = self.preprocess_image(image_path)
            
            # 2. 分类推理
            #print("🔍 分类推理...")
            logits = self.body_inference(model_input)
            
            print(f"✅ 分类 logits: {logits}")

            # 3. 后处理：Softmax 得到概率
            logits_tensor = torch.tensor(logits).unsqueeze(0)  # [1, 2]
            probs = torch.softmax(logits_tensor, dim=1)[0]
            fake_prob = probs[1].item()
            true_prob = probs[0].item()
            #pred_class = 1 if fake_prob > real_prob else 0
           
            
            adaptive_thres = False
            # 5.计算置信度
            if adaptive_thres:
                thres = 0.07
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

            #print(f"🎉 预测结果: {result}")
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
    model_dir = PROJECT_ROOT / "om_models"
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        return False

    required_models = [
        "fusion_expert.om"
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

def Fusion_Expert_Init():
    detector = Fusion_Expert_Ascend(model_dir = PROJECT_ROOT / "om_models")
    return detector
    
def Fusion_Expert_DeInit(detector = None):
    del detector
    print("🧹 资源清理完成")
       
# 项目接口一:Fusion_Expert对输入图像进行推理, 输出图像真假类别和置信
def ISID(detector, test_img):
    print("=" * 60)
    print("🚀 Fusion_Expert_Ascend 推理引擎启动")
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

# 项目接口二:Fusion_Expert对输入视频抽取若干帧进行推理, 综合几帧推理结果，返回视频检测真假结果和平均置信度
def VSID(detector, video_path, num_frames=100, threshold=0.5):
    """
    伪造视频检测主接口
    Args:
        detector: 已初始化的检测器对象
        video_path (str): 视频文件路径
        num_frames (int): 采样帧数，默认 50
        threshold (float): 判定为 fake 的概率阈值，默认 0.5

    Returns:
        dict: {'prediction': 'fake/real', 'confidence': float, 'fake_prob_avg': float}
    """
    print("=" * 60)
    print("🎥 Fusion_Expert 视频伪造检测启动")
    print("=" * 60)

    try:
        # 1. 提取帧
        frames = extract_frames(video_path, num_frames=num_frames)
        if len(frames) == 0:
            print("❌ 未提取到任何有效帧")
            return {'prediction': 'error', 'confidence': 0.0, 'error': 'no frames extracted'}

        fake_probs = []  # 存储每一帧被预测为假的概率
        predictions = []  # 存储每一帧的预测结果（基于阈值）
        confidences = []  # 存储每一帧的置信度

        # 2. 遍历每一帧进行检测
        for idx, frame in enumerate(frames):
            print(f"\n🖼️  处理第 {idx+1}/{len(frames)} 帧...")
            result = detector.predict(frame)  # 注意：这里传入的是 PIL.Image

            # 获取该帧被预测为假的概率
            # 在单帧预测中，fake_prob 对应 probs[1]
            fake_prob = result['confidence'] if result['prediction'] == 'Fake' else (1 - result['confidence'])
            
            fake_probs.append(fake_prob)
            
            # 基于阈值判断单帧结果
            frame_pred = 'Fake' if fake_prob > threshold else 'real'
            predictions.append(frame_pred)
            confidences.append(result['confidence'])

            print(f"   帧 {idx+1} 预测: {frame_pred}, 假概率: {fake_prob:.4f}, 置信度: {result['confidence']:.4f}")

        # 3. 计算平均假概率
        avg_fake_prob = np.mean(fake_probs)
        
        # 4. 基于平均假概率进行最终判断
        final_pred = 'Fake' if avg_fake_prob > threshold else 'real'
        
        # 5. 计算最终置信度
        if final_pred == 'Fake':
            final_confidence = avg_fake_prob
        else:
            final_confidence = 1 - avg_fake_prob  # 真实视频的置信度

        final_result = {
            'prediction': final_pred,
            'confidence': float(final_confidence),
            'fake_prob_avg': float(avg_fake_prob),  # 平均假概率
            'frame_count': len(fake_probs),
            'fake_frame_count': sum(1 for p in predictions if p == 'Fake'),
            'real_frame_count': sum(1 for p in predictions if p == 'real'),
            'per_frame_fake_probs': [float(p) for p in fake_probs],  # 每帧的假概率
            'per_frame_predictions': predictions
        }

        print(f"\n📊 视频检测统计:")
        print(f"   总帧数: {final_result['frame_count']}")
        print(f"   判为假的帧数: {final_result['fake_frame_count']}")
        print(f"   判为真的帧数: {final_result['real_frame_count']}")
        print(f"   平均假概率: {avg_fake_prob:.4f}")
        print(f"📊 视频最终预测结果:")
        print(f"   类别: {final_result['prediction']}")
        print(f"   置信度: {final_result['confidence']:.4f}")
        print(f"   决策阈值: {threshold}")

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


def main():

    #ISID example 1
    #test_img = "test_images/1_fake/example_fake.jpeg"
    #detector = Fusion_Expert_Init()
    #label, confidence = ISID(detector, test_img)
    #Fusion_Expert_DeInit(detector)
    
    #VSID example 2
    detector = Fusion_Expert_Init()
    test_video = "test_videos/reai_3.mp4"  
    final_result = VSID(detector, test_video)  
    Fusion_Expert_DeInit(detector)

if __name__ == "__main__":
    main()