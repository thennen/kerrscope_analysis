import os

directory = os.getcwd()

c = raw_input('Change EVERY space to _ in\n' + directory + '\n?')
if c != 'y':
    raise Exception('Did nothing')

root = []
rename = []
for path, folders, files in os.walk(directory):
    for f in folders + files:
        if ' ' in f:
            root.append(path)
            rename.append(f)

# Go backwards through list so that you rename deeper files first
for d, f in zip(root, rename)[::-1]:
    oldname = os.path.join(d, f)
    print(oldname)
    newname = os.path.join(d, f.replace(' ', '_'))
    os.rename(oldname, newname)
