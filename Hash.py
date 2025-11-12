import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import random
import os

def analyze(file):
    table = pd.read_csv(file, dtype=object)
    hash = lambda x: x % 4999
    histo = []
    for i in range(len(table)):
        try:
            key = int(table.loc[i]['key'])
            histo.append(hash(key))
        except:
            continue
    data = Counter(histo)
    plt.bar(data.keys(), data.values())
    plt.xlabel('hash')
    plt.ylabel('count')
    plt.show()




name = 'commands.csv'

if not os.path.exists(name):
    table = pd.DataFrame(columns=['key', 'command'])

    keys = []
    commands = []

    N = 5000
    M = 100000
    j = 0
    coms = ['select', 'delete']
    for i in range(N):
        key = None
        command = None
        create = (random.randint(1, 5) == 1)
        if create:
            key = int(random.randint(1, M))
            command = random.choice(coms)
            j = j + 1
        keys.append(key)
        commands.append(command)

    print(f"Nonempty rows: {j}")
    table['key'] = keys
    table['key'] = table['key'].astype('Int64')
    table['command'] = commands
    file_name = 'commands.csv'
    table.to_csv(file_name, index=False)

analyze(name)