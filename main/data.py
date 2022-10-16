import wget
import os
import gzip
import shutil
import links


current_dir = os.getcwd()
data_dir = os.path.join(current_dir, 'data')

for link in links.links:
    file_name = os.path.split(link)[-1][:-3]  # without the '.gz' at the end
    
    file_path = os.path.join(data_dir, file_name)
    
    # skip download if file already exists
    if os.path.exists(file_path):
        continue

    # download zipped file
    zipped_file_path = os.path.join(data_dir, f'{file_path}.gz')
    file = wget.download(link, zipped_file_path)

    # unzip file
    temp_file_path = os.path.join(data_dir, f'temp_{file_name}')
    with gzip.open(zipped_file_path, 'rb') as f_in:
        with open(temp_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    with open(temp_file_path, 'r') as inp:
        with open(file_path, 'w') as outp:
            for line in inp:
                line = bytes(line, 'cp1252', 'ignore').decode('utf-8', 'ignore')
                line = line.replace('\"', '')
                outp.write(line)

    # clean up
    try:
        os.remove(zipped_file_path)
        os.remove(temp_file_path)
        os.remove('data/*.tmp')
    except:
        print('Error removing files')
        pass
