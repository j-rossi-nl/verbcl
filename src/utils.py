import glob
import os
import tqdm
import time
import threading
import re

from multiprocessing import Pool


def multiprocess_courtlistener(process_builder, nbworkers=4, monitoring=True):
    """
    Runs a processing task as many parallel tasks, using multiprocessing. The task is to go through all files with
    extension 'file_extension' in folder 'folder'.

    :param process_builder: The builder for the processing function. has to return in_folder, in_file_extension,
                            out_folder, out_file_extension
    :param nbworkers: numbers of parallel processes
    :param monitoring: if True a progress bar will be displayed in a different thread
    :return: a concatenated list of all values returned by the process function
    """
    in_folder, in_file_extension, out_folder, out_file_extension, process = process_builder()
    files = glob.glob(os.path.join(in_folder, '*.{}'.format(in_file_extension)))
    if monitoring:
        output_progress_bar(out_folder, out_file_extension, len(files))
    with Pool(nbworkers) as p:
        results = p.map(process, files)
    return results


def output_progress_bar(folder, file_extension, nb_expected):
    """
    Helps with multiprocessing monitoring.

    :param folder: the folder where the process is outputting data
    :param file_extension: the extension for generated file
    :param nb_expected: number of files we expect to see
    :return: displays a progress bar
    """
    def __output_progress_bar():
        curr_nb_files = len(glob.glob(os.path.join(folder, '*.{}'.format(file_extension))))
        nb_end = curr_nb_files + nb_expected
        with tqdm.tqdm(total=nb_expected) as pbar:
            while curr_nb_files < nb_end:
                time.sleep(1)
                new_nb_files = len(glob.glob(os.path.join(folder, '*.{}'.format(file_extension))))
                if new_nb_files > curr_nb_files:
                    pbar.update(new_nb_files - curr_nb_files)
                    pbar.set_description(time.strftime('%H:%M:%S'))
                curr_nb_files = new_nb_files

    threading.Thread(target=__output_progress_bar).start()


match_extension = re.compile(r'^(?P<nam>.+)(?P<ext>\.\w+)$')


def create_destfilepath(origfile, destpath, addsuffix='', new_extension=None):
    """
    Create a path for a file with same name as origfile but with an added suffix before the file extension
    For example: origfile=/home/juju.csv, destpath=/usr, addsufix='rossi' generates /usr/juju_rossi.csv'

    :param origfile: Path to an original file
    :param destpath: Destination path
    :param addsuffix: a suffix to add
    :param new_extension: if not None, will replace the file extension
    :return: a path for a new file
    """
    origfilename = os.path.basename(origfile)
    if new_extension is None:
        newname = match_extension.sub(r'\g<nam>_{}\g<ext>'.format(addsuffix), origfilename)
    else:
        newname = match_extension.sub(r'\g<nam>_{}.{}'.format(addsuffix, new_extension), origfilename)
    destfullpath = os.path.join(destpath, newname)
    return destfullpath
