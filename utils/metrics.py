import torch
import numpy as np
import clip
from PIL import Image, ImageSequence
from constants import * 

# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git
class Metrics():
    """
    This class constructed to compute the metrics of the model.

    The metrics are:
        - clip_text
        - clip_image
        - pixel_mse

    More information about the metrics can be found in the paper:
        - https://arxiv.org/abs/2104.08718 - CLIPScore: A Reference-free Evaluation Metric for Image Captioning
        - https://openreview.net/pdf?id=bKBhQhPeKaF - Benchmark for Compositional Text-to-Image Synthesis
        - https://arxiv.org/abs/2303.12688 - Pix2Video: Video Editing using Image Diffusion
    """

    def __init__(self, edited_video: np.array, edit_prompt: str) -> None:
        """
        Args:
            edited_video (np.array): edited video T x C x H x W
            edit_prompt (str): edit prompt
        """

        setattr(self, 'device', torch.device("cuda")
                if torch.cuda.is_available() else torch.device("cpu"))
        setattr(self, 'edited_video', edited_video)
        setattr(self, 'edit_prompt', edit_prompt)
        self.model, _ = clip.load("ViT-B/32", device=self.device)

    def compute_pixel_mse_score(self):
        pass

    def compute_clip_text_score(self, edit_prompt: str, edited_frame: np.array) -> float:
        """
        Clip text metric quantifies the similarity between CLIP embedding of edit prompt and CLIP embedding of the edited frame.

        Args:
            edit_prompt (str): edit prompt
            edited_frame (np.array): edited frame

        Returns:
            clip_text_score (float): clip text metric
        """
        with torch.no_grad():
            text_embedding = self.model.encode_text(edit_prompt)
            frame_embedding = self.model.encode_image(edited_frame)



        clip_text_score = torch.cosine_similarity(
            text_embedding, frame_embedding, dim=-1).item()
        del text_embedding, frame_embedding
        return clip_text_score

    def compute_clip_image_score(self, current_frame: np.array, next_frame: np.array) -> float:
        """
        Clip image metric quantifies the similarity between CLIP embedding of current frame and CLIP embedding of the next frame.

        Args:
            current_frame (np.array): current frame
            next_frame (np.array): next frame

        Returns:
            clip_image_score (float): clip image metric
        """
        current_frame_embedding = self.model.encode_image(current_frame)
        next_frame_embedding = self.model.encode_image(next_frame)


        clip_image_score = torch.cosine_similarity(
            current_frame_embedding, next_frame_embedding, dim=-1).item()
        del current_frame_embedding, next_frame_embedding
        return clip_image_score


def main():

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    _, preprocess = clip.load("ViT-B/32", device=device)

    video_dir = f'{DATA_PATH}/car.gif'
    gif = Image.open(video_dir)
    frames = [preprocess(frame).unsqueeze(0) for frame in ImageSequence.Iterator(gif)]

    for idx in range(len(frames)):
        if idx == 0:
            edited_video = torch.tensor(frames[idx])
        else:
            edited_video = torch.cat((edited_video, torch.tensor(frames[idx])), dim=0)

    edited_video = edited_video.to(device)
    
    if edited_video.shape[0] < 2:
        raise Exception('Unsufficent number of frames in the gif.')

    edit_prompt = "a jeep car is moving on the snow"
    edit_prompt = clip.tokenize([edit_prompt]).to(device)

    metrics = Metrics(edited_video=edited_video, edit_prompt=edit_prompt)

    clip_text_score = []
    clip_image_score = []
    for idx, frame in enumerate(metrics.edited_video):
        if idx != 0:
            clip_image_score.append(metrics.compute_clip_image_score(
                current_frame=metrics.edited_video[idx-1].unsqueeze(0), next_frame=frame.unsqueeze(0)))

        clip_text_score.append(metrics.compute_clip_text_score(
            edit_prompt=metrics.edit_prompt, edited_frame=frame.unsqueeze(0)))

    print(f"Clip text score: {np.mean(clip_text_score)}")
    print(f"Clip image score: {np.mean(clip_image_score)}")


if __name__ == '__main__':
    main()