'''
Makes use of Mult-Layer Perceptron model to create custom datasets for different models.

author: Omar Barazanji

Python 3.7.x
'''

import pandas as pd
import os

df = pd.read_csv('dataset.csv')
df = df.drop('Number', 1)
df = df.drop('Action_needed', 1)
df = df.drop('Time', 1)
df = df.drop('Question', 1)


# light application model
df_light = pd.DataFrame([
    ['lights', 'none', 'red', 'set the lights to red'],
    ['lights', 'none', 'red', 'please set the lights to red'],
    ['lights', 'none', 'red', 'set the lights to red'],
    ['lights', 'none', 'red', 'lights to red'],
    ['lights', 'none', 'white', 'can you set the lights to white please'],
    ['lights', 'none', 'white', 'set the lights to white'],
    ['lights', 'none', 'orange', 'please set the lights to orange'],
    ['lights', 'none', 'orange', 'set the lights to orange'],
    ['lights', 'none', 'yellow', 'set the lights to yellow'],
    ['lights', 'none', 'green', 'set the lights to green'],
    ['lights', 'none', 'blue', 'set the lights to blue'],
    ['lights', 'none', 'purple', 'set the lights to purple'],
    ['lights', 'none', 'sparkle', 'set the lights to sparkle'],
    ['lights', 'none', 'gradient', 'set the lights to gradient'],
    ], columns=['Category', 'Subcategory', 'Action', 'Sentence'])

def extract_prompts(filepath):
    with open(filepath, 'r') as f:
        fmt_str = ""
        prompt = ""
        prompts = []
        completion = ""
        for x in f.readlines():
            if "Q:" in x:
                prompt = x.strip("\n")+"\\n".strip('Q: \\n')
                if not prompt == '':
                    prompts.append(prompt)
            elif "A:" in x:
                completion = x.strip("\n")+"\\n"
                fmt_str += '{"prompt": "%s", "completion": "%s"}\n' % (prompt, completion)
    return prompts

conv_prompts = extract_prompts('prompts/conversation-application.txt')
mem_prompts = extract_prompts('prompts/memory-application.txt')
spot_prompts = extract_prompts('prompts/spotify-application.txt')
time_prompts = extract_prompts('prompts/timer-application.txt')

# "other" model
other_arr = []
for prompt in conv_prompts:
    other_arr.append(['other', 'none', 'none', prompt])
for prompt in mem_prompts:
    other_arr.append(['other', 'none', 'none', prompt])
for prompt in spot_prompts:
    other_arr.append(['other', 'none', 'none', prompt])
for prompt in time_prompts:
    other_arr.append(['other', 'none', 'none', prompt])
df_other = pd.DataFrame(other_arr, columns=['Category', 'Subcategory', 'Action', 'Sentence'])


df = df.append(df_light, ignore_index=True).append(df_other, ignore_index=True)
df.to_csv('dataset_ditto.csv', index=False)
