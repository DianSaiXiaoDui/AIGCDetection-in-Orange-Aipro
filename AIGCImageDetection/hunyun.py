import os
from openai import OpenAI
import base64

def hunyun_chat(result = "Fake", image_path = "example.jpeg", confidence = None):
    # 构造 client
    client = OpenAI(
        api_key="sk-uqjqqKTCPxVPYKSq4iTfrh0DXMPhOsVU19xL0k21BzuveWCV",  # 混元 APIKey
        base_url="https://api.hunyuan.cloud.tencent.com/v1",  # 混元 endpoint
    )
    
    print(f"result:{result}, confidence:{confidence}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        image_url = f"data:image/jpeg;base64,{encoded_string}"
    modified_text = ""
    if result == "True":
        modified_text = "你是一个专门研究图像生成的专家,认为这张图片是真实的,请以科学专业的口吻,不要出现任何其他内容,只需要从频率特征和语义特征分析为什么这张图片是真实的,以一段话的形式给出简洁的解释[第一句话点明图片真/假以及置信概率值]"
        if confidence is not None:
           # modified_text += f"严谨起见,你认为这张图片是真实的概率为 {true_prob}"
            modified_text = modified_text.replace("认为这张图片是真实的", f"严谨起见,你认为这张图片是真实的概率为 {confidence:.2f}")
    elif result == "Fake" :
        modified_text = "你是一个专门研究图像生成的专家,认为这张图片是AI生成的,请以科学专业的口吻,不要出现任何其他内容,只需要从频率特征和语义特征分析为什么这张图片是AI生成的,以一段话的形式给出简洁的解释[第一句话点明图片真/假以及置信概率值]"
        if confidence is not None:
            modified_text = modified_text.replace("认为这张图片是AI生成的", f"严谨起见,你认为这张图片是AI生成的概率为 {confidence:.2f}")
    print("prompt: ", modified_text)
    completion = client.chat.completions.create(
        # model="hunyuan-turbos-latest",
        model = "hunyuan-t1-vision-20250916",
        messages=[
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": modified_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                        #"url": "data:image/jpeg;base64,xxxxxxx"
                    }
                }
            ]
            }
        ],
        extra_body={
            "enable_enhancement": True,  # <- 自定义参数
        },
    )
    # completion = client.chat.completions.create(
    # model="hunyuan-vision",
    # messages=[
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "text",
    #                 "text": "What's in this image?"
    #             },
    #             {
    #                 "type": "image_url",
    #                 "image_url": {
    #                     "url": "https://qcloudimg.tencent-cloud.cn/raw/42c198dbc0b57ae490e57f89aa01ec23.png"
    #                     #"url": "data:image/jpeg;base64,xxxxxxx"
    #                 }
    #             }
    #         ]
    #     },
    # ],
    # )
    
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

if __name__ == "__main__":
    hunyun_chat(result = "Fake", image_path = "example_fake.jpeg", confidence = 0.5)