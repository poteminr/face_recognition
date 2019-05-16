class BD_roman:
    def __init__(self):
        self.names = list()
        self.encoded = list()
        
    def add(self, name, vector):
        self.names.append(name)
        self.encoded.append(vector)
    
    def get_names(self):
        return self.names
    
    def get_vectors(self):
        return self.encoded
            
    def get_data(self):
        return self.names, self.encoded