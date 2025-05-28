import gzip
import glob
import os

# Open the gzip file
image = gzip.open('./data/BIDMC/Case00.nii.gz', 'rb')

# Unzip the file and save
for folder in glob.glob('./data/*'): 
    try: 
        os.makedirs(os.path.join(folder, "processed"), exist_ok=True)
        os.makedirs(os.path.join(folder, "original"), exist_ok=True)
    except Exception as e:
        print(f"Error creating directories in {folder}: {e}")
        continue

    for file in glob.glob(os.path.join(folder, "*.gz")):
        try:
            with gzip.open(file, 'rb') as f_in:
                with open(os.path.join(folder, "processed", os.path.basename(file)[:-3]), 'wb') as f_out:
                    f_out.write(f_in.read())
            os.rename(file, os.path.join(folder, "original", os.path.basename(file)))
            print(f"Unzipped {file} to {os.path.join(folder, 'processed', os.path.basename(file)[:-3])}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")