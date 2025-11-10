import asyncio
import edge_tts
import subprocess
import os
import uuid
from pydub import AudioSegment

async def speak_file_offline(file_path, device_name=None, audio_name = None, del_audio=True, read=True):
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    # 合成语音
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    
    # 生成唯一文件名
    if audio_name:
       mp3_name = f'{audio_name}.mp3'
       wav_name = f'{audio_name}.wav'
    else:
       unique_id = uuid.uuid4().hex
       mp3_name = f"audio_{unique_id}.mp3"
       wav_name = f"audio_{unique_id}.mp3"   
    
    mp3_path = os.path.join(script_dir, "audios", mp3_name)
    wav_path = os.path.join(script_dir, "audios", wav_name)
    
    # 保存MP3文件
    await communicate.save(mp3_path)
    
    # 转换为WAV格式
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    
    # 播放WAV文件
    if read:
        if device_name:
            cmd = ['aplay', '-D', device_name, wav_path]
        else:
            cmd = ['aplay', wav_path]
        
        subprocess.run(cmd)
    
    # 清理文件
    os.unlink(mp3_path)
    if del_audio:
        os.unlink(wav_path)
        
    return wav_name   

if __name__ == '__main__':
    asyncio.run(speak_file_offline("test.txt", read=False, del_audio=False, audio_name = "test_0"))