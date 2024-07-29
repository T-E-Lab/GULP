# video.py

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

from moviepy.editor import VideoClip, VideoFileClip, CompositeVideoClip, TextClip
from moviepy.video.io.bindings import mplfig_to_npimage

from gulp2p.preproc.tiff import Tiff


plt.style.use('dark_background')

def show_frame(animation, time=1):
    ax = plt.subplot()
    ax.imshow(animation.get_frame(time))
    ax.set_axis_off()

def grayscale_to_rgb_frame(frame, cmap='viridis'):
    cm = plt.get_cmap(cmap)
    rgb_frame = cm(frame)
    # Cut off alpha channel
    rgb_frame = rgb_frame[:,:,:3] * 255
    return rgb_frame.astype(int)

def get_tiff_frame_at_time(tiff, time, convert_to_rbg=False, motion_correct=True):
    tiff_frame_idx = round(time * tiff.metadata['volume_rate'])
    tiff_vol_count = tiff.metadata['SizeT']
    if tiff_frame_idx > tiff_vol_count:
        tiff_frame_idx = tiff_vol_count - 1
    mip_stack = tiff.get_mip_stack(motion_correct)
    
    frame = mip_stack[tiff_frame_idx]
    if convert_to_rbg:
        frame = grayscale_to_rgb_frame(frame)
    return frame

def make_tiff_frame_creator(tiff, motion_correct=True):
    def make_tiff_frame(time):
        # Given a timepoint return the frame at that time.
        frame = get_tiff_frame_at_time(tiff, time, convert_to_rbg=True, motion_correct=motion_correct)
        return frame
    return make_tiff_frame

def get_frame_near_time(time_series, time):
    return time_series.sub(time).abs().idxmin()

def make_hdpov_frame_creator(expDf, size_px, style="hdvtime"):
    # Used to make frame creater showing fly's head direction as vertical line matching its hd. 
    # convert size to inches for matplot
    px = 1/plt.rcParams['figure.dpi']  # inches per pixel
    size = (size_px[0]*px, size_px[1]*px)
    left = 0.01
    right = 0.01
    top = 0.2
    bottom = 0.35
    fig = plt.figure(figsize=size, layout="constrained")
    ax = fig.add_subplot()
    ax.tick_params(direction='in')

    # hdvtime params
    time_window = 5 # seconds

    # Shift range of angles to center on 0
    expDf['centered_angle'] = np.where(expDf['angle']>180, expDf['angle']-360, expDf['angle'])

    def make_hdpov_frame(time):
        # https://codereview.stackexchange.com/questions/204549/lookup-closest-value-in-pandas-dataframe
        cur_frame_idx = get_frame_near_time(expDf['posTime'], time)
        head_direction = expDf['centered_angle'].iloc[cur_frame_idx]
        if style == "pov":
            ax.axvline(head_direction)
            ax.set_xticks(range(0,360+1,90))
            ax.set_xlim([0, 360])
        if style == "hdvtime":
            # Plot head direction vs time
            old_frame_idx = get_frame_near_time(expDf['posTime'], time-5)
            ax.plot(expDf['posTime'].iloc[cur_frame_idx],
                    expDf['centered_angle'].iloc[cur_frame_idx],
                    marker='o',
                    markersize=1,
                    linestyle='none',
                    color='C0')
            min_time = max(0, time-5)
            max_time = max(5, time)
            ax.set_xlim([min_time, max_time])
            ax.set_ylim([-180,180])
            ax.set_yticks(range(-180, 180+1, 180))
            ax.set_xlabel("time")
            ax.set_title("head direction (deg)",fontsize=9)

        image = mplfig_to_npimage(fig)
        plt.close(fig)
        return image
    return make_hdpov_frame

def make_fly_path_frame_creator(expDf, size_px):
    # convert size to inches for matplot
    px = 1/plt.rcParams['figure.dpi']  # inches per pixel
    size = (size_px[0]*px, size_px[1]*px)
    left = 0.01
    right = 0.01
    top = 0.2
    bottom = 0.35
    fig = plt.figure(figsize=size, layout="constrained")
    ax = fig.add_subplot()

    ax.set_aspect('equal')
    ax.tick_params(direction='in')

    time_window = 5 # seconds

    def make_fly_path_frame(time):
        cur_frame_idx = get_frame_near_time(expDf['posTime'], time)
        old_frame_idx = get_frame_near_time(expDf['posTime'], time-5)
        ax.plot(expDf['x'].iloc[cur_frame_idx-1:cur_frame_idx+1],
                expDf['y'].iloc[cur_frame_idx-1:cur_frame_idx+1],
                marker='none',
                # markersize=1,
                linestyle='-',
                color='C0')
        image = mplfig_to_npimage(fig)
        plt.close(fig)
        return image
    return make_fly_path_frame

def create_video(output_path, tiff_path, synced_bhv_img_path, fictrac_video_path=None, duration=None, fps=60):
    # Read behavioral file
    with open(synced_bhv_img_path, 'rb') as pkl:
        exptDat = pickle.load(pkl)
    expDf = exptDat['expDf']

    # Load tiff
    tiff = Tiff(tiff_path)

    # Create each video panel
    # Parameters
    if duration is None:
        duration = expDf['posTime'].iloc[-1]
    fps = 60 # Frames per second
    is_video_exists = (fictrac_video_path is not None)

    # Fictrac
    fictrac_video_midline = 318 # Pixels
    fictrac_animation = (VideoFileClip(fictrac_video_path.as_posix())
                        .subclip(0,duration)
                        .crop(x1=0, x2=fictrac_video_midline)
                        .margin(left=10))
    # Tiff
    make_tiff_frame = make_tiff_frame_creator(tiff, motion_correct=True)
    tiff_animation = (VideoClip(make_tiff_frame)
                      .resize(height=fictrac_animation.size[1]))

    # Head Direction
    if is_video_exists:
        width = fictrac_animation.size[0]-10
        height = 100
    else:
        width = tiff_animation.size[0]
        height = 100
    make_hd_frame = make_hdpov_frame_creator(expDf,
                                             size_px=(width, height),
                                             style="hdvtime")
    head_direction_animation = (VideoClip(make_hd_frame)
                                .margin(bottom=5, left=10))

    # Fly Path
    # Plot fly path if there is no fictrac video
    if is_video_exists:
        width=fictrac_animation.size[0]-10
        height = 160
    else:
        width = tiff_animation.size[0]
        height = tiff_animation.size[1]-100
    make_fly_path_frame = make_fly_path_frame_creator(expDf,
                                                      size_px=(width, height))
    fly_path_animation = VideoClip(make_fly_path_frame)

    # Stimuli
    # Get unityvr object
    # Figure out type of stimuli
    # Display stimuli

    # Text
    date, line, fly, trial = tiff.path.name.split(".")[0].split("_")
    text = f"{date} {line}\n{fly}_{trial}" # Split name into 2 lines
    txt_clip = (TextClip(txt = text, fontsize = 12, color = 'white')
                .set_duration(duration))


    # Compose video panels together
    if is_video_exists:
        composite_size = (tiff_animation.size[0] + fictrac_animation.size[0],
                        max(tiff_animation.size[1], fictrac_animation.size[1]) + head_direction_animation.size[1])

        midleft = tiff_animation.size[0]/2 - txt_clip.size[0]/2
        midtop = head_direction_animation.size[1]/2 - txt_clip.size[1]/2

        composite_video = CompositeVideoClip([tiff_animation.set_position(('left','bottom')),
                                            txt_clip.set_position((midleft, midtop)),
                                            fictrac_animation.set_position(('right','bottom')),
                                            head_direction_animation.set_position(('right','top')),
                                            fly_path_animation.set_position(('right','bottom'))],
                                            size=composite_size)
        composite_video = composite_video.subclip(0, duration)
    else:
        composite_size = (tiff_animation.size[0]*2,
                        tiff_animation.size[1] + head_direction_animation.size[1])

        midleft = tiff_animation.size[0]/2 - txt_clip.size[0]/2
        midtop = head_direction_animation.size[1]/2 - txt_clip.size[1]/2

        composite_video = CompositeVideoClip([tiff_animation.set_position(('left','bottom')),
                                            txt_clip.set_position((midleft, midtop)),
                                            head_direction_animation.set_position(('right','top')),
                                            fly_path_animation.set_position(('right','bottom'))],
                                            size=composite_size)
        composite_video = composite_video.subclip(0, duration)


    output = Path(f"../results/videos/{tiff_path.stem}.mp4").as_posix()
    composite_video.write_videofile(output_path.as_posix(), fps=fps)
    composite_video.close()
    fictrac_animation.close()
