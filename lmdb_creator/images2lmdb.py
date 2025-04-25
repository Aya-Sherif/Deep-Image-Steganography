import io
import os
import pathlib
import struct
import sys
import time

import PIL.Image
import lmdb
import msgpack

import numpy as np

# Prepare paths (adjust based on your setup)
IMAGE_DIR = r"../tiny-imagenet-200/train"  # Train images path
META_PATH = r"../tiny-imagenet-200/wnids.txt"  # Class IDs (wnids)
WORDS_PATH = r"../tiny-imagenet-200/words.txt"  # Class labels (words)
MDB_OUT_DIR = r"../lmdb_creator"  # Output directory for LMDB

# Set seed for reproducibility
seed = 42
np.random.seed(seed)

lmdb_map_size = 50 * 1024 * 1024 * 1024  # 50 GB
lmdb_txn_size = 500

# Read metadata (wnids.txt and words.txt)
with open(META_PATH, 'r') as f:
    class_ids = f.read().splitlines()

with open(WORDS_PATH, 'r') as f:
    class_labels = f.read().splitlines()

# Prepare metadata for Tiny ImageNet
meta_info = [{
    'ILSVRC2012_ID': idx + 1,  # Create an ID from index
    'WNID': class_ids[idx],
    'words': class_labels[idx],
    'gloss': '',  # Gloss can be left empty if not provided
    'wordnet_height': 0,  # Can be set to 0 as it's not available in Tiny ImageNet
    'num_train_images': 500  # Tiny ImageNet has 500 training images per class
} for idx in range(len(class_ids))]

# Save metadata in msgpack format
META_MP_PATH = os.path.join(MDB_OUT_DIR, 'meta.msgpack')
meta_info_packed = msgpack.packb(meta_info, use_bin_type=True)
with open(META_MP_PATH, 'wb') as f:
    f.write(meta_info_packed)

# LMDB Generation
def make_context():
    return {
        'image_id': 0,
        'clock_beg': time.time(),
        'clock_end': time.time(),
    }

def process_image_one(txn, image_id, wordnet_id, label, image_abspath):
    '''Processes a single image and stores it in LMDB'''
    with PIL.Image.open(image_abspath) as im, io.BytesIO() as bio:
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im = im.resize((256, 256))  # Resize image to 256x256
        im.save(bio, format='WEBP')
        image_bytes = bio.getvalue()

    filename = os.path.basename(image_abspath).rstrip('.JPEG')

    info = {
        'wordnet_id': wordnet_id,
        'filename': filename,
        'image': image_bytes,
        'rows': im.size[1],
        'cols': im.size[0],
        'cnls': 3,  # RGB channels
        'label': label,
    }
    key = '{:08d}'.format(image_id).encode()
    txn.put(key, msgpack.packb(info, use_bin_type=True))

def imagenet_walk(wnid_meta_map, image_Dir):
    '''Walks through the image directory structure and yields image information'''
    def get_category_image_abspaths(Path):
        return [str(f.absolute()) for f in Path.iterdir() if f.is_file()]

    def process_category_one(count, category_Path):
        wordnet_id = category_Path.name
        metainfo = wnid_meta_map[wordnet_id]
        words = metainfo['words']
        gloss = metainfo['gloss']
        label = metainfo['ILSVRC2012_ID']

        print(f'Process count={count}, label={label}, wordnet_id={wordnet_id}')
        print(f'  {words}: {gloss}')
        for image_abspath in get_category_image_abspaths(category_Path):
            yield {
                'label': label,
                'wordnet_id': wordnet_id,
                'image_abspath': image_abspath
            }

    categories = [d for d in image_Dir.iterdir() if d.is_dir()]

    image_files = [
        image_info
        for count, category_Path in enumerate(categories)
        for image_info in process_category_one(count, category_Path)
    ]
    return image_files

def process_images(ctx, lmdb_env, image_infos, image_total):
    '''Processes and writes images to LMDB'''
    image_id = ctx['image_id']

    with lmdb_env.begin(write=True) as txn:
        for image_info in image_infos:
            wordnet_id = image_info['wordnet_id']
            label = image_info['label']
            image_abspath = image_info['image_abspath']
            process_image_one(txn, image_id, wordnet_id, label, image_abspath)
            image_id += 1

    clock_beg = ctx['clock_beg']
    clock_end = time.time()

    elapse = clock_end - clock_beg
    elapse_h = int(elapse) // 60 // 60
    elapse_m = int(elapse) // 60 % 60
    elapse_s = int(elapse) % 60

    estmt = (image_total - image_id) / image_id * elapse
    estmt_h = int(estmt) // 60 // 60
    estmt_m = int(estmt) // 60 % 60
    estmt_s = int(estmt) % 60

    labels = [image_info['label'] for image_info in image_infos]
    print(f'ImageId: {image_id:8d}/{image_total:8d}, time: {elapse_h:2d}h/{elapse_m:2d}m/{elapse_s:2d}s, '
          f'remain: {estmt_h:2d}h/{estmt_m:2d}m/{estmt_s:2d}s, Sample: {str(labels)[:80]} ...')

    ctx['image_id'] = image_id
    ctx['clock_end'] = clock_end

# Prepare metadata mapping
wnid_meta_map = {m['WNID']: m for m in meta_info}

# Open LMDB
image_train_env = lmdb.open(os.path.join(MDB_OUT_DIR, 'tiny_imagenet_train.mdb'), map_size=lmdb_map_size)

# Get image paths and shuffle
image_infos = imagenet_walk(wnid_meta_map, pathlib.Path(IMAGE_DIR))
image_total = len(image_infos)
np.random.shuffle(image_infos)

# Process images in batches
ctx = make_context()
for image_infos_partial in np.array_split(image_infos, lmdb_txn_size):
    process_images(ctx, image_train_env, image_infos_partial, image_total)

print("LMDB file created successfully.")
