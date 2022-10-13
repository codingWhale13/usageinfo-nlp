import wget
import os
import gzip
import shutil
import pandas as pd
import links

for link in links.links:
    file_name=link.split('/')[-1][:-3]
    file_location=f'data/{file_name}'
    if not os.path.exists(file_location):
        url=f'https://s3.amazonaws.com/amazon-reviews-pds/tsv/{file_name}.gz'
        file = wget.download(url, f'{file_location}.gz')

        with gzip.open(f'{file_location}.gz', 'rb') as f_in:
            with open(file_location, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(f'{file_location}.gz')
        os.remove('data/*.tmp')
