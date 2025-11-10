from pathlib import Path
import hunyun as HY
#from TextToSpeechOffline import PaddleSpeechPlayer
import inference_models.SAFE_Series.inference_SAFE as SAFE
import inference_models.SAFE_Series.inference_SR as SAFE_RINE
import inference_models.Fusion_Expert.inference as Fusion_Expert
from EDGETTS import speak_file_offline
import asyncio
import os


SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR

# 香橙派后端服务器ip和端口
ip_addr = '10.181.236.140'
port = 4000

        
def SID(img_path, model = "SAFE", mode = "local", explain = True, speak = True, detector=None, timestamp = None):
  all_result = {}
  # model inference, return class label('True' or 'Fake') and confidence probability
  if model == "SAFE":
     prediction, confidence = SAFE.ISID(detector, img_path) 
  elif model == "SAFE_RINE":
     prediction, confidence = SAFE_RINE.ISID(detector, img_path)   
  elif model == "Fusion_Expert": 
     prediction, confidence = Fusion_Expert.ISID(detector, img_path)
  
  if mode == "web":
     read, del_audio = True, False
  elif mode == "local":
     read, del_audio = True, True
     
  if timestamp: # "Y_H_D_H_M_S"
    audio_name = timestamp
    
  if explain:
    content = HY.hunyun_chat(result=prediction, image_path=img_path, confidence=confidence)
    with open('result.txt', 'w') as f:
      f.write(content)
      
  if speak:
    #player = PaddleSpeechPlayer()    
    #player.speak_file("result.txt")
    import threading

    def speak_file_thread():
        asyncio.run(speak_file_offline("result.txt", read=read, audio_name=audio_name, del_audio=del_audio))

    # 创建并启动一个线程来执行 speak_file_offline，不阻塞主程序
    t = threading.Thread(target=speak_file_thread)
    t.start()
    wav_name = f"{audio_name}.wav"
    audioUrl = f"http://{ip_addr}:{port}/audios/{wav_name}"
    
    
  all_result['category'], all_result['confidence'], all_result['explanation'], all_result['audio_filename'] = prediction, confidence, content, audioUrl 
  return all_result                

if __name__ == '__main__':
    # Locak test, single image
    detector = SAFE.SAFE_Init()
    SID('test_images/0_real/9.jpg', mode="local", explain=True, speak=True, detector= detector, timestamp="2031_11_11_17_06_05" )
    SAFE.SAFE_DeInit(detector)
    
    #_speak = False
    #_explain = False
    '''
    #img = PROJECT_ROOT / "test_images/0_real/9.jpg"
    real_folder_path = PROJECT_ROOT / "test_images/0_real"
    fake_folder_path = PROJECT_ROOT / "test_images/1_fake"
    real_confidence = 0
    fake_confidence = 0
    real_total = 0
    fake_total = 0
    real_result = 0
    fake_result = 0
    for root, dirs, files in os.walk(real_folder_path):
      for file in files:
          # 拼接文件的完整路径
          file_path = os.path.join(root, file)
          category,confidence = SID(file_path, explain = _explain, speak = _speak,detector=detector)
          real_result = real_result + 1 if result == "True" else real_result
          real_total += 1
          real_confidence += confidence
    for root, dirs, files in os.walk(fake_folder_path):
      for file in files:
          # 拼接文件的完整路径
          file_path = os.path.join(root, file)
          category,confidence = SID(file_path, explain = _explain, speak = _speak,detector=detector)
          fake_result = fake_result + 1 if result == "Fake" else fake_result
          fake_total += 1
          fake_confidence += confidence
    SAFE.SAFE_DeInit(detector)
    print(f"真实图像检测成功概率：{real_result/real_total},平均置信度：{real_confidence/real_total}")
    print(f"AI生成图像检测成功概率：{fake_result/fake_total},平均置信度：{fake_confidence/fake_total}")
    '''
    