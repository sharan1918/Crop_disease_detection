import json

with open('model_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('load_folder(os.path.join(DATA_DIR, \'Healthy\'), \'Healthy\')\n' in line for line in source):
            new_source = []
            for line in source:
                new_source.append(line)
                if line == 'load_folder(os.path.join(DATA_DIR, \'Healthy\'), \'Healthy\')\n':
                    new_source.append('load_folder(os.path.join(DATA_DIR, \'Others\'), \'Others\')\n')
            cell['source'] = new_source
            break

with open('model_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("model_training.ipynb updated successfully.")
