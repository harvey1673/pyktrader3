import skpy
import sec_bits
import logging

class SkypeHandler(logging.Handler):
    def __init__(self, user_conf = sec_bits.skype_user):
        # run the regular Handler __init__
        logging.Handler.__init__(self)
        self.user_conf = user_conf
        self.skype_user = skpy.Skype(user_conf['user'], user_conf['pwd'])
        self.group_name = user_conf.get('group_name', 'AlgoTrade')
        self.conn = None
        group_chats = [sc for sc in self.skype_user.chats.recent() if hasattr(sc, 'topic') and sc.topic == self.group_name]
        if len(group_chats)> 0:
            self.conn = group_chats[0]
            return
        group_chats = [sc for sc in self.skype_user.chats if hasattr(sc, 'topic') and sc.topic == self.group_name]
        if len(group_chats)> 0:
            self.conn = group_chats[0]
        else:
            print("couldn't find the group chat from skype")


    def emit(self, record):
        if self.conn:
            self.conn.sendMsg(record.message)