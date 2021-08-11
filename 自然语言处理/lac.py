"""
这是使用paddlehub进行lac分词的代码
"""
import paddlehub as hub
import time
import os, json

model = hub.Module(name='lac')

if __name__ == '__main__':

    if not os.path.exists("res/"):
        os.mkdir('res/')
    while True:
        x = input("请输入要分词的字符串:\n>>")
        x = [x]
        lac_words = model.lexical_analysis(data={"text": x})
        ans = []
        for result in lac_words:
            print(result['word'])
            ans.append(result['word'])

        filename = "res/LAC{}.txt".format(time.time())
        with open(filename, "w") as f:
            f.write(json.dumps(ans, ensure_ascii=False))
        print("已保存到文件 {} ！".format(filename))
