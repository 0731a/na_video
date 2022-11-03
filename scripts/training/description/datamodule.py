import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Annotation:
    def __init__(self, video_name, timestamp, attributes):
        self.video_name = video_name
        self.timestamp = timestamp
        self.emotion = int(attributes["emotion"]) - 1

class DescriptionDataset(Dataset):
    def __init__(self, video_names, directory, video_features_file, max_seq_len=250):
        self.directory = directory
        self.video_names = video_names
        self.video_features_file = video_features_file
        self.max_seq_len = max_seq_len
        self.annotations = self.prefetch_annotations()

        # Because we can't use DDP with IterableDataset,
        # data must be pre-chunked to combat OOM.
        # self.data = self.prefetch_annotations()
        # self.data_size, self.index_to_chunk, self.labels = self.prefetch_and_index()

    def prefetch_annotations(self):
        name_set = set(self.video_names)
        data = {}
        index = 0
        for label_file in tqdm(
            Path(self.directory).glob(f"**/*.json"), desc="Prefetching data..."
        ):
            file_name = label_file.stem      # 예시: [KBS]kim370_대법원 업무 과부하…상고 법원이 대안_18567498.json
            # annotator id 제거하면 비디오 이름 추출.
            # 파일 이름 reverse ([::-1]) 후 "_" 찾음.
            annotator_id_index = len(file_name) - file_name[::-1].find("_") - 1
            video_name = file_name[:annotator_id_index]
            if video_name in name_set:
                with open(label_file, "r", encoding='UTF-8') as rf:
                    json_data = json.load(rf)
                timelines = json_data["timelines"]
                for timeline in timelines:
                    start, end = timeline["start"], timeline["end"]
                    attributes = timeline["attributes"]
                    timestamp = ((end + start) // 2 ) // 10
                    
                    annotation = Annotation(video_name, timestamp, attributes)
                    data[index] = annotation
                    index += 1
        return data

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):

        annotation = self.annotations[index]
        try:
            with h5py.File(self.video_features_file, "r") as vf:
                print("video name : ", annotation.video_name)
                print("timestamp : " , annotation.timestamp)
                video_feature = vf[annotation.video_name][()][annotation.timestamp]
            emotion = int(annotation.emotion)
        except Exception as e:
            # print(f"Corruption in video {annotation.video_name}")
            print(e)
            emotion = 0
            video_feature = torch.rand((3, 224, 224))
        return (
            annotation.video_name,
            video_feature,
            emotion,
        )


if __name__ == "__main__":

    dataDirectory = "E:/042.동영상 콘텐츠 하이라이트 편집 및 설명(요약) 데이터"
    featureFile = "E:/result/video_feature.h5"

    videos = [
        "유튜브_기타_19809",
        "[유튜브_기타_19867"
    ]
    dd = DescriptionDataset(videos, dataDirectory, featureFile)
    dl = DataLoader(dd, batch_size=1)
    for _ in dl:
        pass
