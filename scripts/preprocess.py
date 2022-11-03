import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm
from transformers import ViTFeatureExtractor
import datetime


def extract_video_features(extractor, video_file, sample_every):
    
    start = datetime.datetime.now()

    vc = cv2.VideoCapture(str(video_file))
    fps = int(vc.get(cv2.CAP_PROP_FPS))
    frames = []
    last_collected = -1
    while vc.isOpened():

        success, frame = vc.read()
        if not success:
            break

        timestmap = vc.get(cv2.CAP_PROP_POS_MSEC)
        second = timestmap // 10000
        if second != last_collected:
            last_collected = second
            frames.append(frame)

    features = extractor(images=frames, return_tensors="pt")
    
    print("파일 처리 시간 :: " , datetime.datetime.now() - start)
    
    return features["pixel_values"]


if __name__ == "__main__":
    
    print("전처리 시작 시간 :: ", datetime.datetime.now())

    parser = argparse.ArgumentParser()
    data_directory = "E:/042.동영상 콘텐츠 하이라이트 편집 및 설명(요약) 데이터/01.데이터"
    out = "E:/result/video_feature.h5"
    sample_every = -1
    args = parser.parse_args()

    video_files = tqdm(list(Path(data_directory).glob("**/*.mp4")))
    extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224", size=224)

    with h5py.File(out, "w") as wf:

        for video_file in video_files:
            name = video_file.stem
            try:
                features = extract_video_features(
                    extractor, video_file, sample_every=sample_every
                )
                wf.create_dataset(name, data=features)
            except Exception as e:
                print(e)


    print("전처리 종료 시간 :: ", datetime.datetime.now())