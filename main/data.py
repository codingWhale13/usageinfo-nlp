import wget
import os
import gzip
import shutil
import links

for link in links.links:
    file_name=link.split('/')[-1][:-3]
    file_location=f'data/{file_name}'
    if not os.path.exists(file_location):
        file = wget.download(link, f'{file_location}.gz')

        with gzip.open(f'{file_location}.gz', 'rb') as f_in:
            with open(f'data/temp_{file_name}', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        with open(f'data/temp_{file_name}', 'r') as inp:
            with open(file_location, 'w') as outp:
                for line in inp:
                    line=bytes(line, 'cp1252', 'ignore').decode('utf-8','ignore')
                    line=line.replace('\"', '')
                    outp.write(line)

        try:
            os.remove(f'{file_location}.gz')
            os.remove(f'data/temp_{file_name}')
            os.remove('data/*.tmp')
        except:
            print('Error removing files')
            pass
