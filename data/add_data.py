import pandas as pd
import os
df = pd.read_csv('dataset.csv')
df = df.drop('Number', 1)
df = df.drop('Action_needed', 1)
df = df.drop('Time', 1)
df = df.drop('Question', 1)

df_test = pd.DataFrame([
    ['lights', 'none', 'red', 'set the lights to red'],
    ['lights', 'none', 'white', 'set the lights to white'],
    ['lights', 'none', 'orange', 'set the lights to orange'],
    ['lights', 'none', 'yellow', 'set the lights to yellow'],
    ['lights', 'none', 'green', 'set the lights to green'],
    ['lights', 'none', 'blue', 'set the lights to blue'],
    ['lights', 'none', 'purple', 'set the lights to purple'],
    ['lights', 'none', 'sparkle', 'set the lights to sparkle'],
    ['lights', 'none', 'gradient', 'set the lights to gradient'],
    ], columns=['Category', 'Subcategory', 'Action', 'Sentence'])

df = df.append(df_test, ignore_index=True)

df.to_csv('dataset_new.csv', index=False)