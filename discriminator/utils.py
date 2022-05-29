import copy, json
from threading import Lock

class ProbaseClient:
    def __init__(self):
        from multiprocessing.connection import Client
        address = ('localhost', 9233)
        print('Connecting Probase...')
        self.conn = Client(address)
        print('Connected')
        self.cache = {}
        self._lock = Lock()

    def query(self, x, target='abs', sort_method='mi', truncate=50):
        x = x.lower()
        if (x, sort_method, truncate) in self.cache:
            return copy.copy(self.cache[(x, sort_method, truncate)])
        with self._lock:
            self.conn.send(json.dumps([x, sort_method, truncate, target]))
            res = json.loads(self.conn.recv())
            key_remove = []
            for key in res:
                if key.split(' ')[-1] in ['word', 'phrase', 'noun', 'adjective', 'verb', 'pronoun', 'term', 'aux']:
                    key_remove.append(key)
            for k in key_remove:
                del res[k]
            self.cache[(x, sort_method, truncate)] = copy.copy(res)
        return res