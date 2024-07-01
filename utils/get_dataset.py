import pandas as pd
from pathlib import Path
import gdown
import zipfile
import shutil
from sklearn.model_selection import train_test_split

_URL = "https://drive.google.com/uc?id=1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj"
_URL_LABELS = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels/labels.csv"

if __name__=="__main__":
    data_root = Path('../data/').absolute()
    download_path = data_root / 'downloads'
    images_download = download_path / 'images.zip'

    images_path = data_root / 'images'
    train_path = images_path / 'train'
    test_path = images_path / 'test'

    seed = 42

    image_zip_path = gdown.cached_download(_URL, images_download, quiet=True)

    labels_df = pd.read_csv(_URL_LABELS)

    # Create a test split
    train_df, test_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df.Label, random_state=seed)
    train_df['split'] = 'train'
    train_df['base_path'] = train_path
    test_df['split'] = 'test'
    test_df['base_path'] = test_path
    labels_df = pd.concat([train_df, test_df])

    # Figure out the directories to save each image in following the 
    # tf.keras.preprocessing.image_dataset_from_directory expected format
    labels_df['species'] = labels_df.Species.str.replace(' ', '_').str.lower()
    labels_df['label_path'] = ''

    for label in labels_df['species'].unique():
        for split_path in (train_path, test_path):
            label_path = split_path / label
            shutil.rmtree(label_path, ignore_errors=True)
            label_path.mkdir(parents=True)
            labels_df.loc[(labels_df.species == label) & (labels_df.base_path == split_path), 'label_path'] = label_path


    labels_df['file_path'] = labels_df.label_path / labels_df.Filename

    path_matches_label = labels_df.apply(lambda row: row.file_path.parent.name == row.species, axis=1)
    assert path_matches_label.all(), "Somehow we've messed up saving some images in their proper label directory"

    # Extract each image to the correct train / test label folder
    with zipfile.ZipFile(image_zip_path, 'r') as image_zip:
        files = image_zip.namelist()
        labelled_files = labels_df.Filename
        for file in labelled_files:
            assert file in files

        extract_img = lambda row: image_zip.extract(row.Filename, row.label_path)
        labels_df.apply(extract_img, axis=1)
    labels_df.to_csv(data_root / 'labels.csv', index=False)