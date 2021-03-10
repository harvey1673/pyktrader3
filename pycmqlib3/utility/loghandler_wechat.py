import itchat
import logging

class WechatHandler(logging.Handler):  # Inherit from logging.Handler
    def __init__(self, user_conf = {'nickName': 'harvey'}):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        itchat.auto_login()
        self.user_conf = user_conf
        self.user_name = None
        users = []
        if 'nickName' in user_conf:
            users = itchat.search_friends(**user_conf)
        elif 'name' in user_conf:
            users = itchat.search_chatrooms(**user_conf)

        if len(users) <= 0:
            print("unsupported config")
            return
        self.user_name = users[0].userName

    def emit(self, record):
        if self.user_name:
            itchat.send(record.message, self.user_name)

