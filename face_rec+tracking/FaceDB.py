import json
import time
import os
class FileDB:
    def __init__(self, file):

        self.file_name = file
        self.data = {"actions":[], "action_count":0}

        if os.path.exists(self.file_name) == False:
            json_text = json.dumps(self.data)
            
            self.f = open(self.file_name, "w+")
            self.f.write(json.dumps(self.data))
            self.f.close()

    def append_action(self, cam, action):
        t = time.localtime()

        action_timed = {"time":time.asctime(t),"cam":str(cam), "action":action}

        self.data["actions"].append(action_timed)
        self.data["action_count"] = len(self.data["actions"])
        
        json_text = json.dumps(self.data)

        self.f = open(self.file_name, "w+")
        self.f.write(json_text)
        self.f.close()

    def read_data(self):

        self.f = open(self.file_name, "r")
        json_test = self.f.read()
        self.f.close()
        
        if len(json_test) < 2:
            json_text = json.dumps(self.data)
            
            self.f = open(self.file_name, "w+")
            self.f.write(json_text)
            self.f.close()
        else:
            self.data = json.loads(json_test)

