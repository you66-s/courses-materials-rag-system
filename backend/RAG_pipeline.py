import os
from data_loader import DataLoader

raw_data = os.listdir('data')
for folder in raw_data:
    folder_path = os.path.join('data', folder)
    files = os.listdir(folder_path)
    for file in files:
        file_path = os.path.join(folder_path, file)
        modules = os.listdir(file_path)
        for module in modules:
            material_path = os.path.join(file_path, module)
            documents = DataLoader(file_path=material_path, reading_mode="single").load_data()
            print(f"Loaded {len(documents)} documents from {module}")
            