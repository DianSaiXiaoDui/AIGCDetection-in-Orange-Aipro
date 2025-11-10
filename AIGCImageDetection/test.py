import os
import tempfile
import subprocess
import logging
import sys
from importlib import import_module

# 修复 aistudio_sdk 导入问题
try:
    from aistudio_sdk.hub import download
except ImportError:
    # 尝试手动导入
    try:
        hub_module = import_module('aistudio_sdk.hub')
        download = hub_module.download
    except Exception as e:
        print(f"无法导入 aistudio_sdk.hub.download: {e}")
        # 创建一个假的 download 函数
        def download(*args, **kwargs):
            raise RuntimeError("aistudio_sdk.hub.download 不可用")
        sys.modules['aistudio_sdk.hub'].download = download
from paddlespeech.cli.tts import TTSExecutor

class PaddleSpeechPlayer:
    """PaddleSpeech 文本朗读播放器"""
    
    def __init__(self, model_name="fastspeech2_csmsc", vocoder_name="pwgan_csmsc"):
        """
        初始化 PaddleSpeech 播放器
        
        参数:
            model_name: TTS 模型名称 (默认: fastspeech2_csmsc)
            vocoder_name: 声码器名称 (默认: pwgan_csmsc)
        """
        self.model_name = model_name
        self.vocoder_name = vocoder_name
        self.logger = logging.getLogger("PaddleSpeechPlayer")
        self.logger.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # 初始化 TTS 执行器
        try:
            self.tts_executor = TTSExecutor()
            self.logger.info("PaddleSpeech 播放器初始化完成")
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            raise
    
    def list_audio_devices(self):
        """列出可用的音频设备"""
        try:
            # Linux 系统
            if os.name == 'posix':
                result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
                self.logger.info("可用音频设备:\n" + result.stdout)
                return result.stdout
            # Windows 系统
            elif os.name == 'nt':
                result = subprocess.run(
                    ['powershell', '-Command', 'Get-WmiObject -Class Win32_SoundDevice | Select-Object Name, Status'],
                    capture_output=True, text=True
                )
                self.logger.info("音频设备信息:\n" + result.stdout)
                return result.stdout
            else:
                self.logger.warning("无法列出音频设备: 不支持的操作系统")
                return "无法列出音频设备: 不支持的操作系统"
        except Exception as e:
            self.logger.error(f"列出音频设备失败: {e}")
            return f"错误: {e}"
    
    def _find_device_index(self, device_name):
        """根据设备名称查找设备索引"""
        try:
            if os.name == 'posix':  # Linux
                result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if device_name in line:
                        # 提取设备索引 (格式: card X: ...)
                        parts = line.split()
                        for part in parts:
                            if part.startswith('card'):
                                return part.split(':')[0].replace('card', '').strip()
            return None
        except Exception as e:
            self.logger.error(f"查找设备索引失败: {e}")
            return None
    
    def _play_audio(self, audio_file, device_name=None):
        """播放音频文件"""
        try:
            if device_name:
                # 查找设备索引
                device_index = self._find_device_index(device_name)
                if device_index:
                    # 使用指定设备播放
                    cmd = ['aplay', '-D', f'plughw:{device_index}', audio_file]
                else:
                    # 直接使用设备名称
                    cmd = ['aplay', '-D', device_name, audio_file]
            else:
                # 使用默认设备
                cmd = ['aplay', audio_file]
            
            self.logger.info(f"播放命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"播放失败: {result.stderr}")
                return False
            
            self.logger.info("播放完成")
            return True
            
        except Exception as e:
            self.logger.error(f"播放音频失败: {e}")
            return False
    
    def synthesize_speech(self, text, output_file=None, device_name=None):
        """
        合成语音并播放
        
        参数:
            text: 要合成的文本
            output_file: 输出文件路径（可选）
            device_name: 音频设备名称（可选）
        """
        try:
            # 如果没有指定输出文件，创建临时文件
            if output_file is None:
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                output_file = temp_file.name
                temp_file.close()
                is_temp_file = True
            else:
                is_temp_file = False
            
            self.logger.info(f"开始合成语音: {text[:50]}...")
            
            # 使用 PaddleSpeech 合成语音
            self.tts_executor(
                text=text,
                output=output_file,
                am=self.model_name,
                voc=self.vocoder_name
            )
            
            self.logger.info(f"语音合成完成: {output_file}")
            
            # 播放音频
            success = self._play_audio(output_file, device_name)
            
            # 清理临时文件
            if is_temp_file:
                os.unlink(output_file)
            
            return success
            
        except Exception as e:
            self.logger.error(f"语音合成失败: {e}")
            # 清理临时文件
            if is_temp_file and os.path.exists(output_file):
                os.unlink(output_file)
            return False
    
    def speak_file(self, file_path, device_name=None, encoding='utf-8'):
        """
        朗读文本文件内容
        
        参数:
            file_path: 文本文件路径
            device_name: 音频设备名称（可选）
            encoding: 文件编码（默认: utf-8）
        """
        try:
            # 检查文件是否存在
            if not os.path.isfile(file_path):
                error_msg = f"文件不存在: {file_path}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # 读取文件内容
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text_content = f.read().strip()
            except UnicodeDecodeError:
                # 尝试其他编码
                with open(file_path, 'r', encoding='gbk') as f:
                    text_content = f.read().strip()
            
            if not text_content:
                error_msg = "文件内容为空"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self.logger.info(f"开始朗读文件: {file_path}")
            return self.synthesize_speech(text_content, device_name=device_name)
            
        except Exception as e:
            self.logger.error(f"朗读文件失败: {e}")
            return False
    
    def save_speech_to_file(self, text, output_file):
        """
        将语音保存到文件（不播放）
        
        参数:
            text: 要合成的文本
            output_file: 输出文件路径
        """
        try:
            self.logger.info(f"将语音保存到文件: {output_file}")
            
            # 使用 PaddleSpeech 合成语音
            self.tts_executor(
                text=text,
                output=output_file,
                am=self.model_name,
                voc=self.vocoder_name
            )
            
            self.logger.info(f"语音已保存到: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存语音文件失败: {e}")
            return False

# 使用示例
if __name__ == "__main__":
    # 创建播放器实例
    player = PaddleSpeechPlayer()
    
    # 列出可用音频设备
    #player.list_audio_devices()
    
    # 测试直接朗读文本
    player.synthesize_speech("你好，这是一个PaddleSpeech语音合成测试")
    
    # 测试朗读文件
    player.speak_file("test.txt")
    
    # 测试使用指定设备播放
    #player.speak_file("test.txt", "USB PnP Sound Device")
    
    # 测试保存语音到文件
    #player.save_speech_to_file("这是保存的语音", "output.wav")