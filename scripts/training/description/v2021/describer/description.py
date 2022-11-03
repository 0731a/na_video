import argparse

from transformers import ViTFeatureExtractor

from v2021 import extract_video_features

from .model import DescriptionModel


class Describer:
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device
        self.video_feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.model = DescriptionModel.load_from_checkpoint(model_path).to(self.device)

    def describe(self, video_path):
        video_features = extract_video_features(
            self.video_feature_extractor, video_path
        ).to(self.model.device)
        emotion = self.model(video_features)
        emotion = emotion.argmax(dim=-1).mode().values.item()
        return emotion


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--argument", help="Example argument")
    args = parser.parse_args()
