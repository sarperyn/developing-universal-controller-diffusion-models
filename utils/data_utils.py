import decord
import ffmpeg
decord.bridge.set_bridge('torch')

# from torch.utils.data import Dataset


def return_video(video_path, device, n_sample_frames=None, sample_start_idx=0, sample_frame_rate=1):
    vr = decord.VideoReader(video_path)

    sample_index = list(range(sample_start_idx, len(vr), sample_frame_rate))
    if n_sample_frames is not None:
        sample_index = sample_index[:n_sample_frames]

    video = vr.get_batch(sample_index)
    return (video / 127.5 - 1.0).to(device)

def frames_to_video(frames_folder_path='outputs/frames-2/*.png', output_path='outputs/video_results/edited-man-skiing.mp4'):
    ffmpeg.input(frames_folder_path, pattern_type='glob', framerate=25).output(output_path).run()


# class VideoDataset(Dataset):
#     def __init__(self, video_params):
#         self.video = return_video(**video_params)
        
#         pass
    
#     def __len__(self):
#         return self.video.size(0)-1
    

#     def __getitem__(self, index):
#         pass

#     def get_consecutives(self):
#         cons_frames = list(map(lambda idx: (self.video[idx], self.video[idx+1]), range(len(self.video)-1)))
#         print(cons_frames)

if __name__ == "__main__":
    frames_to_video()