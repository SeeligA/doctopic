import os
import shutil
import logging

from sources.doctopic import update_index_path


def move_from_temp(src, dst):
    files = os.listdir(src)

    for file in files:
        fp = os.path.join(src, file)
        fp_new = os.path.join(dst, file)

        # Look for Similarity indices in contrast to corpus indices
        if file.endswith('lsi.index') or file.endswith('tfidf.index'):
            update_index_path(fp, fp_new)

        else:
            # Copy file to model parameters dir
            shutil.copy2(fp, fp_new)

        logging.info('Copying {} to {}'.format(file, dst))
        # Remove file in temporary folder
        os.unlink(fp)

    return len(files)


def make_model_archive(dst):

    basename = os.path.join(dst)
    root = os.path.dirname(dst)
    basedir = os.path.basename(dst)

    # Create ZIP archive in root directory
    shutil.make_archive(basename, 'zip', root, basedir)

    # Remove files in basedir
    [os.unlink(os.path.join(dst, file)) for file in os.listdir(dst)]

    return root
