from multiprocessing.connection import Listener
import json
from conceptualize_proposer import Proposer
from build_graph_utils import pb_query_abstract, pb_query_instance
import traceback
import datetime

print('Starting server...')
proposer = Proposer()

address = ('localhost', 9233)     # family is deduced to be 'AF_INET'
while True:
    listener = Listener(address)
    print('Server started')
    conn = listener.accept()
    print('Connection accepted from', listener.last_accepted)
    while True:
        try:
            msg = conn.recv()
            if msg is None:
                break
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M '), msg)
            if not isinstance(msg, str):
                conn.send('')
                continue
            msg = json.loads(msg)
            if msg[-1] == 'abs':
                response = pb_query_abstract(proposer, msg[0])
            elif msg[-1] == 'inst':
                response = pb_query_instance(proposer, msg[0])
            elif msg[-1] == 'count':
                response = {'concept_freq': proposer.concept_freq_map.get(msg[0], 0),
                            'inst_freq': proposer.inst_freq_map.get(msg[0], 0),
                            'entity_freq': proposer.entity_freq_map.get(msg[0], 0),
                            'inst_cnt': proposer.inst_cnt_map.get(msg[0], 0)}
            else:
                raise ValueError(msg)
            print('Collected probase response of %d items' % len(response))
            if msg[1] != '' and msg[-1] != 'count':
                response = list(response.items())
                response.sort(key=lambda x: -x[1][msg[1]])
                if msg[2] != -1:
                    response = response[:msg[2]]
                response = dict(response)
            response = json.dumps(response)
            print('Send %d bytes' % len(response))
            conn.send(response)
        except:
            traceback.print_exc()
            break

    listener.close()
