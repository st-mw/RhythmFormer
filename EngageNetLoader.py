""" A dataloader for EngageNet. Adapted from dataset/data_loader/PURELoader.py """
import glob
import os

import cv2
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm

import csv

class EngageNetLoader(BaseLoader):
    """ Data loader for the EngageNet dataset."""

    def __init__(self, name, data_path, config_data):
        """Initializes an EngageNet dataloader.
            Args:
                data_path(str): path of a folder which stores raw videos as .mp4.
                name(str): name of the dataloader.
                config_data(CfgNode): data settings(ref:config.py).
        """
        super().__init__(name, data_path, config_data)
        
    # Overriding __getitem__ from baseloader for rppg extraction only
    def __getitem__(self, index):
        """Returns a clip of video(3,T,W,H) and its corresponding signals(T)."""
    
        data = np.load(self.inputs[index])

        print("***** data.shape in __getitem__****")
        print(data.shape)

        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format == 'NDHWC':
            pass
        else:
            raise ValueError('Unsupported Data Format!')

        data = np.float32(data)

        # item_path is the location of a specific clip in a preprocessing output folder
        # For example, an item path could be /home/data/PURE_SizeW72_...unsupervised/501_input0.npy
        item_path = self.inputs[index]
        # item_path_filename is simply the filename of the specific clip
        # For example, the preceding item_path's filename would be 501_input0.npy
        item_path_filename = item_path.split(os.sep)[-1]

        # split_idx represents the point in the previous filename where we want to split the string 
        # in order to retrieve a more precise filename (e.g., 501) preceding the chunk (e.g., input0)
        split_idx = item_path_filename.rindex('_')
        # Following the previous comments, the filename for example would be 501
        filename = item_path_filename[:split_idx]
        # chunk_id is the extracted, numeric chunk identifier. Following the previous comments, 
        # the chunk_id for example would be 0
        chunk_id = item_path_filename[split_idx + 6:].split('.')[0]

        return data, filename, chunk_id    

    def get_raw_data(self, data_path):
        """ Returns list of video file paths """

        data_files = glob.glob(os.path.join(data_path, "*.mp4"))
        if not data_files:
            raise ValueError(self.dataset_name + " data paths empty!")

        return data_files

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        frames = list()
        cap = cv2.VideoCapture(video_file)

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"FPS of video {video_file} = {fps}")

        if fps > 35:
            print(f"Incompatible frame rate: {video_file}, FPS = {fps}")

            selected_frames = EngageNetLoader.convert_fps(video_file, fps)
            return selected_frames

        if not cap.isOpened():
            print(f"Error: Unable to open video file {video_file}")
            return np.array([])

        try: 
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            cap.release()

        except Exception as e:
            print(f"Error reading video file {video_file}: {e}")   
        finally:
            cap.release()     

        print(f"in read_video: Finished reading {video_file} - Total frames: {len(frames)}")
        return np.asarray(frames)

    def split_raw_data(self, data_dirs, begin, end):
        # Overrides super: Not splitting data for extract only.
        # Empty function to override ValueError in super
        return data_dirs

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
       filename_w_ext = os.path.split(data_dirs[i])[-1]
       
       saved_filepathname = data_dirs[i] 

       filename_no_ext = os.path.splitext(filename_w_ext)[0]

       # Can add options here for data aug. See PURELoader.py.
       frames = self.read_video(saved_filepathname)

       bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)


       target_length = frames.shape[0]
       bvps = BaseLoader.resample_ppg(bvps, target_length)
       frames_clips = self.preprocess(frames, bvps, config_preprocess)
       frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)

       input_name_list, label_name_list = self.save_multi_process(frames_clips=frames_clips, bvps_clips=bvps_clips, filename=filename_no_ext)
       file_list_dict[i] = input_name_list
       
    def save_multi_process(self, frames_clips, bvps_clips=None, filename=None):
        """Save all the chunked data with multi-thread processing.

        Args:
            frames_clips(np.array): blood volumne pulse (PPG) labels.
            bvps_clips(np.array): the length of each chunk.
            filename: name the filename
        Returns:
            input_path_name_list: list of input path names
            label_path_name_list: list of label path names
        """
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        count = 0
        input_path_name_list = []
        label_path_name_list = []
        for i in range(len(frames_clips)): #originally len(bvps_clips)
            assert (len(self.inputs) == len(self.labels))
            input_path_name = self.cached_path + os.sep + "{0}_input{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            input_path_name_list.append(input_path_name)
            label_path_name_list.append(label_path_name)
            np.save(input_path_name, frames_clips[i])
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return input_path_name_list, label_path_name_list    

    def preprocess(self, frames, bvps=None, config_preprocess=None):
        # config_preprocess defaults to None only so that bvps can default to None.
        """Overrides super: Preprocesses video data.

        Args:
            frames(np.array): Frames in a video.
            bvps(np.array): Blood volumne pulse (PPG) signal labels for a video.
            config_preprocess(CfgNode): preprocessing settings(ref:config.py).
        Returns:
            frame_clips(np.array): processed video data by frames
            bvps_clips(np.array): processed bvp (ppg) labels by frames
        """
        # resize frames and crop for face region
        frames = self.crop_face_resize(
            frames,
            config_preprocess.CROP_FACE.DO_CROP_FACE,
            config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
            config_preprocess.CROP_FACE.LARGE_BOX_COEF,
            config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
            config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
            config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
            config_preprocess.RESIZE.W,
            config_preprocess.RESIZE.H)
        # Check data transformation type
        data = list()  # Video data
        for data_type in config_preprocess.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        data = np.concatenate(data, axis=-1)  # concatenate all channels
        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        if config_preprocess.DO_CHUNK:  # chunk data into snippets
            frames_clips, bvps_clips = self.chunk(
                data, bvps, chunk_length=config_preprocess.CHUNK_LENGTH)
        else:
            frames_clips = np.array([data])
            bvps_clips = np.array([bvps])

        return frames_clips, bvps_clips

    def chunk(self, frames, bvps=None, chunk_length=None):
        """ Chunk the data into small chunks.

        Args:
            frames(np.array): video frames.
            bvps(np.array): blood volumne pulse (PPG) labels.
            chunk_length(int): the length of each chunk.
        Returns:
            frames_clips: all chunks of face cropped frames
            bvp_clips: all chunks of bvp frames
        """

        clip_num = frames.shape[0] // chunk_length
        frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        bvps_clips = [bvps[i * chunk_length:(i + 1) * chunk_length] for i in range(clip_num)]
        return np.array(frames_clips),  np.array(bvps_clips)

    @staticmethod
    def convert_fps(video_file, fps):
        selected_frames = list()
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            print(f"Error: unable to open {video_file}") 
            return np.ones(300) 

        count_to = fps // 30
        counter = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if counter == count_to:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                selected_frames.append(frame)
                counter = 0
            else:
                counter += 1

        cap.release()
        return np.asarray(selected_frames)

