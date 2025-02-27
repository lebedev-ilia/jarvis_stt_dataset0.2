import yaml
import os
import datetime
import pickle
import json
from dataset.jarvis_stt_dataset.configs.const import const

const = const()

path2config = const.path2config
path2logs = const.path2logs


def logging_used_voice(datadir, ratio_for_next_iter):

    with open(f'{datadir}/{path2config}') as f:
        data = yaml.safe_load(f).get('model')

    manifest_data = [data.get('train_ds').get('manifest_filepath'), data.get('validation_ds').get('manifest_filepath')]

    ids = {'train':[],'validation':[]}

    i = 0

    for m in manifest_data:
        with open(m, 'r') as f:
            m = os.path.split(m)[1]
            key = m[m.index('stt')+4:m.index('_manifest')]
            for el in f:
                ids[key].append(os.path.split(el[el.index('audio_filepath')+17:el.index('duration')-4])[1])
                i += 1

    date = str(datetime.datetime.now())[:-7]
    if not os.path.exists(f'{datadir}/{path2logs}/{date}'):
        os.mkdir(f'{datadir}/{path2logs}/{date}')
        with open(f'{datadir}/{path2logs}/{date}/ids_log.p', 'a'):
            pass
        with open(f'{datadir}/{path2logs}/{date}/ratio_for_next_iter.json', 'a'):
            pass
        
    with open(f'{datadir}/{path2logs}/{date}/ids_log.p', 'wb') as f:      
        pickle.dump(ids, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{datadir}/{path2logs}/{date}/ratio_for_next_iter.json', 'w') as f:      
        json.dump(ratio_for_next_iter, f)
        
    print(f'Logging {i} used voice to ---| logs/{date} |---')
        
        
def get_used_voice(datadir):
    
    log_data = os.listdir(f'{datadir}/{path2logs}')
    
    if len(log_data) == 0:
        print('Empty logs')
    else:
        
        full_path_list = []    
        for log in log_data:
            
            with open(f'{datadir}/{path2logs}/{log}/ids_log.p', 'rb') as f:
                data = pickle.load(f)
                for k in data.keys():
                    
                    voice_list = data.get(k)
                    
                    for el in voice_list:
                        if 'ru' in el:
                            name = el[el.index('ru'):el.index('ru')+2]
                        elif 'en' in el:
                            name = el[el.index('en'):el.index('en')+2]
                        elif 'main' in el:
                            name = el[el.index('main')+5:el.index('.wav')-6]
                            if name[-1] == '_':
                                name[-1].replace('_', '')
                        
                        full_path_list.append(el)

        return list(set(full_path_list))
        


