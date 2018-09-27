from wxpy import *
from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
# from wxpy import get_wechat_logger
import sys
# parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(r'C:\Users\Zack\AppData\Roaming\Python\Python36\site-packages\mitie')

# 之前训练好的NLU模型
config_dir = "config.yml"
nlu_model_dir = "models/nlu/nlu"
dia_model_dir = 'models/dialogue'
domain_file = "domain.yml"

nlu_model_path = nlu_model_dir
# agent = Agent.load("../models/policy/mom", interpreter=RasaNLUInterpreter(nlu_model_path))
agent = Agent.load(dia_model_dir,
                   interpreter=RasaNLUInterpreter(nlu_model_dir))
# 初始化机器人，扫码登陆
bot = Bot(console_qr=False, cache_path=True)

# bot = Bot(console_qr=False, cache_path=True)

# bot.self.add()
# bot.self.accept()
# bot.self.send('哈咯~')
# bot.file_helper.send('哈咯~')
# logger = get_wechat_logger()

# 自动接受新的好友请求
@bot.register(msg_types=FRIENDS)
def auto_accept_friends(msg):
    # 接受好友请求
    new_friend = msg.card.accept()
    # 向新的好友发送消息
    new_friend.send('哈哈，我自动接受了你的好友请求')

# 回复 my_friend 的消息 (优先匹配后注册的函数!)
# @bot.register(my_friend)
# def reply_my_friend(msg):
#     return 'received: {} ({})'.format(msg.text, msg.type)

@bot.register(bot.self, except_self=False)
def reply_self(msg):
    # agent = Agent.load("models/policy/current",
    #                    interpreter=RasaNLUInterpreter(nlu_model_path))
    ans = agent.handle_message(msg.text)
    print(ans)
    return ans[0]['text']
    # return " ".join(ans)
    # return 'received: {} ({})'.format(msg.text, msg.type)

@bot.register(bot.friends())
def reply_my_friend(msg):
    ans = agent.handle_message(msg.text)
    print(ans)
    return ans[0]['text']
    # return " ".join(ans)

# 进入 Python 命令行、让程序保持运行
embed()