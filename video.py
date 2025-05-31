"""Generating or processing video, often using ffmpeg"""
from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import str
from builtins import map
from builtins import input
from builtins import object
import numpy as np
import subprocess
import re
import datetime
import os
import matplotlib.pyplot as plt
import my.plot
try:
    import ffmpeg
except ImportError:
    pass

class OutOfFrames(BaseException):
    """Exception raised when more frames cannot be extracted from a video"""
    pass

def ffmpeg_frame_string(filename, frame_time=None, frame_number=None):
    """Given a frame time or number, create a string for ffmpeg -ss.
    
    This attempts to reverse engineer the way that ffmpeg converts frame
    times to frame numbers, so that we can specify an exact frame number
    and get that exact frame.
    
    As far as I can tell, if you request time T, 
    ffmpeg rounds T to the nearest millisecond, 
    and then gives you frame N, 
    where N is ceil(T * frame_rate).
    
    So -.001 gives you the first frame, and .001 gives you the second frame.
    
    It's hard to predict what will happen within one millisecond of a
    frame time, so try to avoid that if exactness is important.
    
    
    filename : video file. Used to get frame rate.
    
    frame_time : This one takes precedence if both are provided.
        We simply subtract half of the frame interval, and then round to
        the nearest millisecond to account for ffmpeg's rounding up.
    
    frame_number : This one is used if frame_time is not specified.
        We convert to a frame time using
            ((frame_number / frame_rate) - 1 ms) 
            rounded down to the nearest millisecond.
        This should give accurate results as long as frame rate is not
        >500 fps or so.
    
    frametime, frame_number : which frame to get
        if you request time T, ffmpeg gives you frame N, where N is 
        ceil(time * frame_rate). So -.001 gives you the first frame, and
        .001 gives you the second frame. It's hard to predict what will
        happen with one ms of the exact frame time due to rounding errors.
    
    Returns : string, suitable for -ss
    """
    if frame_number is not None:
        # If specified by number, convert to time
        frame_rate = get_video_params(filename)[2]
        use_frame_time = (frame_number / float(frame_rate)) - .001
        use_frame_time = np.floor(use_frame_time * 1000) / 1000.
    
    elif frame_time is not None:
        frame_rate = get_video_params(filename)[2]
        use_frame_time = frame_time - (1. / (2 * frame_rate))
    
    else:
        raise ValueError("must specify frame by time or number")
    
    use_frame_string = '%0.3f' % use_frame_time
    return use_frame_string

def get_frame(filename, frametime=None, frame_number=None, frame_string=None,
    pix_fmt='gray', bufsize=10**9, path_to_ffmpeg='ffmpeg', vsync='drop',
    n_frames=1):
    """Returns a single frame from a video as an array.
    
    This creates an ffmpeg process and extracts data from it with a pipe.

    This syntax is used to seek with ffmpeg:
        ffmpeg -ss %frametime% -i %filename% ...
    This is supposed to be relatively fast while still accurate.
    
    Parameters
    ----------
    filename : video filename
    
    frame_string : to pass to -ss
    
    frametime, frame_number:
        If frame_string is None, then these are passed to 
        ffmpeg_frame_string to generate a frame string.
        
    pix_fmt : the "output" format of ffmpeg.
        currently only gray and rgb24 are accepted, because I need to 
        know how to reshape the result.
    
    n_frames : int
        How many frames to get
    
    
    Returns
    -------
    tuple: (frame_data, stdout, stderr)
        frame : numpy array
            Generally the shape is (n_frames, height, width, n_channels)
            Dimensions of size 1 are squeezed out
        
        stdout : typically blank
        
        stderr : ffmpeg's text output
    """
    v_width, v_height = get_video_aspect(filename)
    
    if pix_fmt == 'gray':
        bytes_per_pixel = 1
        reshape_size = (n_frames, v_height, v_width)
    elif pix_fmt == 'rgb24':
        bytes_per_pixel = 3
        reshape_size = (n_frames, v_height, v_width, 3)
    else:
        raise ValueError("can't handle pix_fmt:", pix_fmt)
    
    # Generate a frame string if we need it
    if frame_string is None:
        frame_string = ffmpeg_frame_string(filename, 
            frame_time=frametime, frame_number=frame_number)
    
    # Create the command
    command = [path_to_ffmpeg, 
        '-ss', frame_string,
        '-i', filename,
        '-vsync', vsync,
        '-vframes', str(n_frames),
        '-f', 'image2pipe',
        '-pix_fmt', pix_fmt,
        '-vcodec', 'rawvideo', '-']
    
    # To store result
    res_l = []
    frames_read = 0

    # Init the pipe
    # We set stderr to PIPE to keep it from writing to screen
    # Do this outside the try, because errors here won't init the pipe anyway
    pipe = subprocess.Popen(command, 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
        bufsize=bufsize)

    try:
        # Read
        read_size = bytes_per_pixel * v_width * v_height * n_frames
        raw_image = pipe.stdout.read(read_size)    
        
        # Raise if not enough data
        if len(raw_image) < read_size:
            raise OutOfFrames        
        
        # Convert to numpy
        flattened_im = np.fromstring(raw_image, dtype='uint8')
        
        # Reshape
        frame_data = flattened_im.reshape(reshape_size)    
        
        # Squeeze if n_frames == 1
        if n_frames == 1:
            frame_data = frame_data[0]
    
    except OutOfFrames:
        print("warning: cannot get frame")
        frame_data = None
    
    finally:
        # Restore stdout
        pipe.terminate()

        # Keep the leftover data and the error signal (ffmpeg output)
        stdout, stderr = pipe.communicate()    
        
        # Convert to string
        if stdout is not None:
            stdout = stdout.decode('utf-8')
        if stderr is not None:
            stderr = stderr.decode('utf-8')
    
    return frame_data, stdout, stderr


def frame_dump(filename, frametime, output_filename='out.png', 
    meth='ffmpeg fast', subseek_cushion=20., verbose=False, dry_run=False,
    very_verbose=False):
    """Dump the frame in the specified file.
    
    Probably better to use get_frame instead.
    
    If the subprocess fails, CalledProcessError is raised.
    Special case: if seek is beyond the end of the file, nothing is done
    and no error is raised
    (because ffmpeg does not report any problem in this case).
    
    Values for meth:
        'ffmpeg best' : Seek quickly, then accurately
            ffmpeg -y -ss :coarse: -i :filename: -ss :fine: -vframes 1 \
                :output_filename:
        'ffmpeg fast' : Seek quickly
            ffmpeg -y -ss :frametime: -i :filename: -vframes 1 :output_filename:
        'ffmpeg accurate' : Seek accurately, but takes forever
            ffmpeg -y -i :filename: -ss frametime -vframes 1 :output_filename:
        'mplayer' : This takes forever and also dumps two frames, the first 
            and the desired. Not currently working but something like this:
            mplayer -nosound -benchmark -vf framestep=:framenum: \
                -frames 2 -vo png :filename:
    
    Note that output files are always overwritten without asking.
    
    With recent, non-avconv versions of ffmpeg, it appears that 'ffmpeg fast'
    is just as accurate as 'ffmpeg best', and is now the preferred method.
    
    Use scipy.misc.imread to read them back in.
    
    Source
        https://trac.ffmpeg.org/wiki/Seeking%20with%20FFmpeg
    """
    
    if meth == 'mplayer':
        raise ValueError("mplayer not supported")
    elif meth == 'ffmpeg best':
        # Break the seek into a coarse and a fine
        coarse = np.max([0, frametime - subseek_cushion])
        fine = frametime - coarse
        syscall = 'ffmpeg -y -ss %r -i %s -ss %r -vframes 1 %s' % (
            coarse, filename, fine, output_filename)
    elif meth == 'ffmpeg accurate':
        syscall = 'ffmpeg -y -i %s -ss %r -vframes 1 %s' % (
            filename, frametime, output_filename)
    elif meth == 'ffmpeg fast':
        syscall = 'ffmpeg -y -ss %r -i %s -vframes 1 %s' % (
            frametime, filename, output_filename)
    
    if verbose:
        print(syscall)
    if not dry_run:
        #os.system(syscall)
        syscall_l = syscall.split(' ')
        syscall_result = subprocess.check_output(syscall_l, 
            stderr=subprocess.STDOUT)
        if very_verbose:
            print(syscall_result)

def process_chunks_of_video(filename, n_frames, func='mean', verbose=False,
    frame_chunk_sz=1000, bufsize=10**9,
    image_w=None, image_h=None, pix_fmt='gray',
    finalize='concatenate', path_to_ffmpeg='ffmpeg', vsync='drop'):
    """Read frames from video, apply function, return result
    
    Uses a pipe to ffmpeg to load chunks of frame_chunk_sz frames, applies
    func, then stores just the result of func to save memory.
    
    If n_frames > # available, returns just the available frames with a
    warning.
    
    filename : file to read
    n_frames : number of frames to process
        if None or np.inf, will continue until video is exhausted
    func : function to apply to each frame
        If 'mean', then func = lambda frame: frame.mean()
        If 'keep', then func = lambda frame: frame
        'keep' will return every frame, which will obviously require a lot
        of memory.
    verbose : If True, prints out frame number for every chunk
    frame_chunk_sz : number of frames to load at once from ffmpeg
    bufsize : sent to subprocess.Popen
    image_w, image_h : width and height of video in pxels
    pix_fmt : Sent to ffmpeg
    """
    if n_frames is None:
        n_frames = np.inf
    
    # Default function is mean luminance
    if func == 'mean':
        func = lambda frame: frame.mean()
    elif func == 'keep':
        func = lambda frame: frame
    elif func is None:
        raise ValueError("must specify frame function")
    
    # Get aspect
    if image_w is None:
        image_w, image_h = get_video_aspect(filename)
    
    # Set up pix_fmt
    if pix_fmt == 'gray':
        bytes_per_pixel = 1
        reshape_size = (image_h, image_w)
    elif pix_fmt == 'rgb24':
        bytes_per_pixel = 3
        reshape_size = (image_h, image_w, 3)
    else:
        raise ValueError("can't handle pix_fmt:", pix_fmt)
    read_size_per_frame = bytes_per_pixel * image_w * image_h
    
    # Create the command
    command = [path_to_ffmpeg,
        '-i', filename,
        '-vsync', vsync,
        '-f', 'image2pipe',
        '-pix_fmt', pix_fmt,
        '-vcodec', 'rawvideo', '-']
    
    # To store result
    res_l = []
    frames_read = 0

    # Init the pipe
    # We set stderr to PIPE to keep it from writing to screen
    # Do this outside the try, because errors here won't init the pipe anyway
    # Actually, stderr will fill up and the process will hang
    # http://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python/4896288#4896288
    pipe = subprocess.Popen(command, 
        stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'), 
        bufsize=bufsize)

    # Catch any IO errors and restore stdout
    try:
        # Read in chunks
        out_of_frames = False
        while frames_read < n_frames and not out_of_frames:
            if verbose:
                print(frames_read)
            # Figure out how much to acquire
            if frames_read + frame_chunk_sz > n_frames:
                this_chunk = n_frames - frames_read
            else:
                this_chunk = frame_chunk_sz
            
            # Read this_chunk, or as much as we can
            raw_image = pipe.stdout.read(read_size_per_frame * this_chunk)
            
            # check if we ran out of frames
            if len(raw_image) < read_size_per_frame * this_chunk:
                #print("warning: ran out of frames")
                out_of_frames = True
                this_chunk = len(raw_image) // read_size_per_frame
                assert this_chunk * read_size_per_frame == len(raw_image)
            
            # Process
            flattened_im = np.fromstring(raw_image, dtype='uint8')
            if bytes_per_pixel == 1:
                video = flattened_im.reshape(
                    (this_chunk, image_h, image_w))
            else:
                video = flattened_im.reshape(
                    (this_chunk, image_h, image_w, bytes_per_pixel))
            
            # Store as list to avoid dtype and shape problems later
            #chunk_res = np.asarray(map(func, video))
            chunk_res = list(map(func, video))
            
            # Store
            res_l.append(chunk_res)
            
            # Update
            frames_read += this_chunk

    except:
        raise

    finally:
        # Restore stdout
        pipe.terminate()

        # Keep the leftover data and the error signal (ffmpeg output)
        stdout, stderr = pipe.communicate()
        
        # Convert to string
        if stderr is not None:
            stderr = stderr.decode('utf-8')

    if not np.isinf(n_frames) and frames_read != n_frames:
        # This usually happens when there's some rounding error in the frame
        # times
        # But it can also happen if more frames are requested than length
        # of video
        # So just warn, not error
        print("warning: requested {} frames but only read {}".format(
            n_frames, frames_read))

    # Stick chunks together
    if len(res_l) == 0:
        print("warning: no data found")
        res = np.array([])
    elif finalize == 'concatenate':
        res = np.concatenate(res_l)
    elif finalize == 'listcomp':
        res = np.array([item for sublist in res_l for item in sublist])
    elif finalize == 'list':
        res = res_l
    else:
        print("warning: unknown finalize %r" % finalize)
        res = res_l
        
    return res

def get_video_aspect(video_filename):
    """Returns width, height of video using ffmpeg-python"""
    if not os.path.exists(video_filename):
        raise ValueError("%s does not exist" % video_filename)
    
    probe = ffmpeg.probe(video_filename)
    assert len(probe['streams']) == 1
    width = probe['streams'][0]['width']
    height = probe['streams'][0]['height']
    
    return width, height

def get_video_frame_rate(video_filename):
    """Returns frame rate of video using ffmpeg-python
    
    https://video.stackexchange.com/questions/20789/ffmpeg-default-output-frame-rate
    """
    if not os.path.exists(video_filename):
        raise ValueError("%s does not exist" % video_filename)
    
    probe = ffmpeg.probe(video_filename)
    assert len(probe['streams']) == 1
    
    # Seems to be two ways of coding, not sure which is better
    avg_frame_rate = probe['streams'][0]['avg_frame_rate']
    r_frame_rate = probe['streams'][0]['r_frame_rate']
    assert avg_frame_rate == r_frame_rate
    
    # Convert fraction to number
    num, den = avg_frame_rate.split('/')
    frame_rate = float(num) / float(den)
    
    return frame_rate

def get_video_params(video_filename):
    """Returns width, height, frame_rate of video using ffmpeg-python"""
    
    width, height = get_video_aspect(video_filename)
    frame_rate = get_video_frame_rate(video_filename)
    return width, height, frame_rate

def get_video_duration(video_filename):
    """Returns duration of a video file
    
    Uses ffmpeg.probe to probe the file, and extracts the duration of the
    container. Checks that the video file contains only a single stream,
    whose duration matches the container's.
    
    Returns : float
        The duration in seconds
        This seems to be exact, so there will be int(np.rint(duration * rate))
        frames in the video. You can request one less than this number using
        my.video.get_frame to get the last frame (Pythonic). If you request 
        this number or more, you will get an error.
    """
    ## Check
    # Check it exists
    if not os.path.exists(video_filename):
        raise ValueError("%s does not exist" % video_filename)
    
    # Probe it
    probe = ffmpeg.probe(video_filename)
    
    # Check that it contains only one stream
    assert len(probe['streams']) == 1
    
    
    ## Container duration
    # This is the easiest one to extract, but in theory the stream duration
    # could differ
    container_duration = float(probe['format']['duration'])
    
    
    ## Stream duration
    if 'DURATION' in probe['streams'][0]['tags']:
        # This tends to be the right way for most ffmpeg-encoded videos
        stream_duration_s = probe['streams'][0]['tags']['DURATION']
        
        # For some reason this is in nanoseconds, convert to microseconds
        stream_duration_s = stream_duration_s[:-3]
        
        # Match
        video_duration_temp = datetime.datetime.strptime(
            stream_duration_s, '%H:%M:%S.%f')
        stream_duration_dt = datetime.timedelta(
            hours=video_duration_temp.hour, 
            minutes=video_duration_temp.minute, 
            seconds=video_duration_temp.second,
            microseconds=video_duration_temp.microsecond)    
        
        # Convert to seconds
        stream_duration = stream_duration_dt.total_seconds()
    else:
        # This works for mjpeg videos from white matter
        stream_duration_s = probe['streams'][0]['duration']
        
        # Convert to seconds
        stream_duration = float(stream_duration_s)
    
    
    ## Check that container and stream duration are the same
    assert stream_duration == container_duration
    
    # Return the single duration
    return stream_duration

def choose_rectangular_ROI(vfile, n_frames=4, interactive=False, check=True,
    hints=None):
    """Displays a subset of frames from video so the user can specify an ROI.
    
    If interactive is False, the frames are simply displayed in a figure.
    If interactive is True, a simple text-based UI allows the user to input
    the x- and y- coordinates of the ROI. These are drawn and the user has
    the opportunity to confirm them.
    
    If check is True, then the values are swapped as necessary such that
    x0 < x1 and y0 < y1.
    
    Finally the results are returned as a dict with keys x0, x1, y0, y1.
    
    hints : dict, or None
        If it has key x0, x1, y0, or y1, the corresponding values will
        be displayed as a hint to the user while selecting.
    """
    import matplotlib.pyplot as plt
    import my.plot
    # Not sure why this doesn't work if it's lower down in the function
    if interactive:
        plt.ion() 

    # Get frames
    duration = get_video_duration(vfile)
    frametimes = np.linspace(duration * .1, duration * .9, n_frames)
    frames = []
    for frametime in frametimes:
        frame, stdout, stderr = get_frame(vfile, frametime)
        frames.append(frame)
    
    # Plot them
    f, axa = plt.subplots(1, 4, figsize=(11, 2.5))
    f.subplots_adjust(left=.05, right=.975, bottom=.05, top=.975)
    for frame, ax in zip(frames, axa.flatten()):
        my.plot.imshow(frame, ax=ax, axis_call='image', cmap=plt.cm.gray)
    my.plot.harmonize_clim_in_subplots(fig=f, clim=(0, 255))

    # Get interactive results
    res = {}
    if interactive:
        params_l = ['x0', 'x1', 'y0', 'y1']
        lines = []
        try:
            while True:
                for line in lines:
                    line.set_visible(False)    
                plt.draw()
                
                # Get entries for each params
                for param in params_l:
                    # Form request string, using hint if available
                    hint = None
                    if hints is not None and param in hints:
                        hint = hints[param]
                    if hint is None:
                        request_s = 'Enter %s: ' % param
                    else:
                        request_s = 'Enter %s [hint = %d]: ' % (param, hint)
                    
                    # Keep getting input till it is valid
                    while True:
                        try:
                            val = input(request_s)
                            break
                        except ValueError:
                            print("invalid entry")
                    res[param] = int(val)

                # Check ordering
                if check:
                    if res['x0'] > res['x1']:
                        res['x0'], res['x1'] = res['x1'], res['x0']
                    if res['y0'] > res['y1']:
                        res['y0'], res['y1'] = res['y1'], res['y0']

                # Draw results
                for ax in axa:
                    lines.append(ax.plot(
                        ax.get_xlim(), [res['y0'], res['y0']], 'r-')[0])
                    lines.append(ax.plot(
                        ax.get_xlim(), [res['y1'], res['y1']], 'r-')[0])
                    lines.append(ax.plot(
                        [res['x0'], res['x0']], ax.get_ylim(), 'r-')[0])            
                    lines.append(ax.plot(
                        [res['x1'], res['x1']], ax.get_ylim(), 'r-')[0])
                plt.draw()

                # Get confirmation
                choice = input("Confirm [y/n/q]: ")
                if choice == 'q':
                    res = {}
                    print("cancelled")
                    break
                elif choice == 'y':
                    break
                else:
                    pass
        except KeyboardInterrupt:
            res = {}
            print("cancelled")
        finally:
            plt.ioff()
            plt.close(f)
    
    return res    


def crop(input_file, output_file, crop_x0, crop_x1, 
    crop_y0, crop_y1, crop_stop_sec=None, vcodec='mpeg4', quality=2, 
    overwrite=True, verbose=False, very_verbose=False):
    """Crops the input file into the output file"""
    # Overwrite avoid
    if os.path.exists(output_file) and not overwrite:
        raise ValueError("%s already exists" % output_file)
    
    # Set up width, height and origin of crop zone
    if crop_x0 > crop_x1:
        crop_x0, crop_x1 = crop_x1, crop_x0
    if crop_y0 > crop_y1:
        crop_y0, crop_y1 = crop_y1, crop_y0
    width = crop_x1 - crop_x0
    height = crop_y1 - crop_y0
    
    # Form the syscall
    crop_string = '"crop=%d:%d:%d:%d"' % (width, height, crop_x0, crop_y0)
    syscall_l = ['ffmpeg', '-i', input_file, '-y',
        '-vcodec', vcodec,
        '-q', str(quality),
        '-vf', crop_string]
    if crop_stop_sec is not None:
        syscall_l += ['-t', str(crop_stop_sec)]
    syscall_l.append(output_file)

    # Call, redirecting to standard output so that we can catch it
    if verbose:
        print(' '.join(syscall_l))
    
    # I think when -t parameter is set, it raises CalledProcessError
    #~ syscall_result = subprocess.check_output(syscall_l, 
        #~ stderr=subprocess.STDOUT)
    #~ if very_verbose:
        #~ print syscall_result
    os.system(' '.join(syscall_l))

def split():
    # ffmpeg -i 150401_CR1_cropped.mp4 -f segment -vcodec copy -reset_timestamps 1 -map 0 -segment_time 1000 OUTPUT%d.mp4
    pass


class WebcamController(object):
    def __init__(self, device='/dev/video0', output_filename='/dev/null',
        width=320, height=240, framerate=30,
        window_title='webcam', image_controls=None,
        ):
        """Init a new webcam controller for a certain webcam.
        
        image_controls : dict containing controls like gain, exposure
            They will be set to reasonable defaults if not specified.
        """
        # Store params
        self.device = device
        self.output_filename = output_filename
        self.width = width
        self.height = height
        self.framerate = framerate
        self.window_title = window_title
        
        if self.output_filename is None:
            self.output_filename = '/dev/null'
        
        # Image controls
        self.image_controls = {
            'gain': 3,
            'exposure': 20,
            'brightness': 40,
            'contrast': 50,
            'saturation': 69,
            'hue': 0,
            'white_balance_automatic': 0,
            'gain_automatic': 0,
            'auto_exposure': 1, # flipped
            }
        
        # Above are for the PS3 Eye
        # This is for C270
        self.image_controls = {
            'gain': 3,
            'exposure_absolute': 1000,
            'brightness': 40,
            'contrast': 30,
            'saturation': 69,
            'white_balance_temperature_auto': 0,
            'exposure_auto': 1,
            }            
        if image_controls is not None:
            self.image_controls.update(image_controls)
        
        self.read_stderr = None
        self.ffplay_stderr = None
        self.ffplay_stdout = None
        
        self.ffplay_proc = None
        self.read_proc = None
        self.tee_proc = None
    
    def start(self, print_ffplay_proc_stderr=False, print_read_proc_stderr=False):
        """Start displaying and encoding
        
        To stop, call the stop method, or close the ffplay window.
        In the latter case, it will keep reading from the webcam until
        you call cleanup or delete the object.
        
        print_ffplay_proc_stderr : If True, prints the status messages to
            the terminal from the the process that plays video to the screen.
            If False, writes to /dev/null.
        print_read_proc_stderr : Same, but for the process that reads from
            the webcam.
        """
        # Set the image controls
        self.set_controls()
        
        # Create a process to read from the webcam
        # stdin should be pipe so it doesn't suck up keypresses (??)
        # stderr should be null, so pipe doesn't fill up and block
        # stdout will go to downstream process
        if print_read_proc_stderr:
            read_proc_stderr = None
        else:
            read_proc_stderr = open(os.devnull, 'w')
        read_proc_cmd_l = ['ffmpeg',
            '-f', 'video4linux2',
            '-i', self.device,
            '-vcodec', 'libx264',
            '-qp', '0',
            '-vf', 'format=gray',
            '-preset', 'ultrafast',
            '-f', 'rawvideo', '-',
            ] 
        self.read_proc = subprocess.Popen(read_proc_cmd_l, stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, stderr=read_proc_stderr)
        
        # Sometimes the read_proc fails because the device is busy or "Input/ouput error"
        # but the returncode isn't set or anything so I don't know how to
        # detect this.

        # Tee the compressed output to a file
        self.tee_proc = subprocess.Popen(['tee', self.output_filename], 
            stdin=self.read_proc.stdout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Play the output
        if print_ffplay_proc_stderr:
            ffplay_proc_stderr = None
        else:
            ffplay_proc_stderr = open(os.devnull, 'w')        
        self.ffplay_proc = subprocess.Popen([
            'ffplay', 
            #~ '-fflags', 'nobuffer', # not compatible with analyzeduration or probesize?
            '-analyzeduration', '500000', # 500 ms delay in starting
            '-window_title', self.window_title,
            '-',
            ], 
            stdin=self.tee_proc.stdout,
            stdout=subprocess.PIPE, stderr=ffplay_proc_stderr)

        # This is supposed to allow SIGPIPE
        # https://docs.python.org/2/library/subprocess.html#replacing-shell-pipeline
        self.read_proc.stdout.close()
        self.tee_proc.stdout.close()        
    
    def set_controls(self):
        """Use v4l2-ctl to set the controls"""
        # Form the param list
        cmd_list = ['v4l2-ctl',
            '-d', self.device,
            '--set-fmt-video=width=%d,height=%d' % (self.width, self.height),
            '--set-parm=%d' % self.framerate,    
            ]
        for k, v in list(self.image_controls.items()):
            cmd_list += ['-c', '%s=%d' % (k, v)]

        # Create a process to set the parameters and run it
        self.set_proc = subprocess.Popen(cmd_list,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.set_stdout, self.set_stderr = self.set_proc.communicate()

        if self.set_proc.returncode != 0:
            print("failed to set parameters")
            print(self.set_stdout)
            print(self.set_stderr)
            raise IOError("failed to set parameters")
    
    def stop(self):
        if self.ffplay_proc is not None:
            self.ffplay_proc.terminate()
        self.cleanup()
    
    def update(self):
        pass
    
    def cleanup(self):
        self.__del__()
    
    def __del__(self):
        if self.ffplay_proc is not None:
            if self.ffplay_proc.returncode is None:
                self.ffplay_stdout, self.ffplay_stderr = \
                    self.ffplay_proc.communicate()
        
        if self.read_proc is not None:
            if self.read_proc.returncode is None:
                self.read_proc.terminate()
                self.read_proc.wait()
        
        if self.tee_proc is not None:
            self.tee_proc.wait()


class WebcamControllerFFplay(WebcamController):
    """Simpler version that just plays with ffplay"""
    def start(self):
        self.set_controls()
        self.ffplay_proc = subprocess.Popen([
            'ffplay',
            '-f', 'video4linux2',
            '-window_title', self.window_title,
            '-i', self.device,
            ], 
            stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'),
            bufsize=1000000)
        self.stdout_l = []
        self.stderr_l = []

    def stop(self):
        self.ffplay_proc.terminate()
        self.cleanup()
    
    def update(self):
        """This is supposed to read the stuff on stderr but I can't
        get it to not block"""
        return
        #~ self.stdout_l.append(self.ffplay_proc.stdout.read())
        print("update")
        data = self.ffplay_proc.stderr.read(1000000)
        print("got data")
        print(len(data))
        while len(data) == 1000000:
            self.stderr_l.append(data)
            data = self.ffplay_proc.stderr.read(1000000)
        print("done")
    
    def __del__(self):
        try:
            if self.ffplay_proc.returncode is None:
                self.ffplay_stdout, self.ffplay_stderr = (
                    self.ffplay_proc.communicate())
        except AttributeError:
            pass


## These were copied in from WhiskiWrap, use these from now on
class FFmpegReader(object):
    """Reads frames from a video file using ffmpeg process"""
    def __init__(self, input_filename, pix_fmt='gray', bufsize=10**9,
        duration=None, start_frame_time=None, start_frame_number=None,
        write_stderr_to_screen=False, vsync='drop'):
        """Initialize a new reader
        
        input_filename : name of file
        pix_fmt : used to format the raw data coming from ffmpeg into
            a numpy array
        bufsize : probably not necessary because we read one frame at a time
        duration : duration of video to read (-t parameter)
        start_frame_time, start_frame_number : -ss parameter
            Parsed using my.video.ffmpeg_frame_string
        write_stderr_to_screen : if True, writes to screen, otherwise to
            /dev/null
        """
        self.input_filename = input_filename
    
        # Get params
        self.frame_width, self.frame_height, self.frame_rate = \
            get_video_params(input_filename)
        
        # Set up pix_fmt
        if pix_fmt == 'gray':
            self.bytes_per_pixel = 1
        elif pix_fmt == 'rgb24':
            self.bytes_per_pixel = 3
        else:
            raise ValueError("can't handle pix_fmt:", pix_fmt)
        self.read_size_per_frame = self.bytes_per_pixel * \
            self.frame_width * self.frame_height
        
        # Create the command
        command = ['ffmpeg']
        
        # Add ss string
        if start_frame_time is not None or start_frame_number is not None:
            ss_string = ffmpeg_frame_string(input_filename,
                frame_time=start_frame_time, frame_number=start_frame_number)
            command += [
                '-ss', ss_string]
        
        command += [
            '-i', input_filename,
            '-vsync', vsync,
            '-f', 'image2pipe',
            '-pix_fmt', pix_fmt]
        
        # Add duration string
        if duration is not None:
            command += [
                '-t', str(duration),]
        
        # Add vcodec for pipe
        command += [
            '-vcodec', 'rawvideo', '-']
        
        # To store result
        self.n_frames_read = 0

        # stderr
        if write_stderr_to_screen:
            stderr = None
        else:
            stderr = open(os.devnull, 'w')

        # Init the pipe
        # We set stderr to null so it doesn't fill up screen or buffers
        # And we set stdin to PIPE to keep it from breaking our STDIN
        self.ffmpeg_proc = subprocess.Popen(command, 
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=stderr, 
            bufsize=bufsize)

    def iter_frames(self):
        """Yields one frame at a time
        
        When done: terminates ffmpeg process, and stores any remaining
        results in self.leftover_bytes and self.stdout and self.stderr
        
        It might be worth writing this as a chunked reader if this is too
        slow. Also we need to be able to seek through the file.
        """
        # Read this_chunk, or as much as we can
        while(True):
            raw_image = self.ffmpeg_proc.stdout.read(self.read_size_per_frame)

            # check if we ran out of frames
            if len(raw_image) != self.read_size_per_frame:
                self.leftover_bytes = raw_image
                self.close()
                return
        
            # Convert to array
            flattened_im = np.fromstring(raw_image, dtype='uint8')
            if self.bytes_per_pixel == 1:
                frame = flattened_im.reshape(
                    (self.frame_height, self.frame_width))
            else:
                frame = flattened_im.reshape(
                    (self.frame_height, self.frame_width, self.bytes_per_pixel))

            # Update
            self.n_frames_read = self.n_frames_read + 1

            # Yield
            yield frame
    
    def close(self):
        """Closes the process"""
        # Need to terminate in case there is more data but we don't
        # care about it
        # But if it's already terminated, don't try to terminate again
        if self.ffmpeg_proc.returncode is None:
            self.ffmpeg_proc.terminate()
        
            # Extract the leftover bits
            self.stdout, self.stderr = self.ffmpeg_proc.communicate()
        
        return self.ffmpeg_proc.returncode
    
    def isclosed(self):
        if hasattr(self.ffmpeg_proc, 'returncode'):
            return self.ffmpeg_proc.returncode is not None
        else:
            # Never even ran? I guess this counts as closed.
            return True

class FFmpegWriter(object):
    """Writes frames to an ffmpeg compression process"""
    def __init__(self, output_filename, frame_width, frame_height,
        output_fps=30, vcodec='libx264', qp=15, preset='medium',
        input_pix_fmt='gray', output_pix_fmt='yuv420p', 
        write_stderr_to_screen=False):
        """Initialize the ffmpeg writer
        
        output_filename : name of output file
        frame_width, frame_height : Used to inform ffmpeg how to interpret
            the data coming in the stdin pipe
        output_fps : frame rate
        input_pix_fmt : Tell ffmpeg how to interpret the raw data on the pipe
            This should match the output generated by frame.tostring()
        output_pix_fmt : pix_fmt of the output
        crf : quality. 0 means lossless
        preset : speed/compression tradeoff
        write_stderr_to_screen :
            If True, writes ffmpeg's updates to screen
            If False, writes to /dev/null
        
        With old versions of ffmpeg (jon-severinsson) I was not able to get
        truly lossless encoding with libx264. It was clamping the luminances to
        16..235. Some weird YUV conversion? 
        '-vf', 'scale=in_range=full:out_range=full' seems to help with this
        In any case it works with new ffmpeg. Also the codec ffv1 will work
        but is slightly larger filesize.
        """
        # Open an ffmpeg process
        cmdstring = ('ffmpeg', 
            '-y', '-r', '%d' % output_fps,
            '-s', '%dx%d' % (frame_width, frame_height), # size of image string
            '-pix_fmt', input_pix_fmt,
            '-f', 'rawvideo',  '-i', '-', # raw video from the pipe
            '-pix_fmt', output_pix_fmt,
            '-vcodec', vcodec,
            '-qp', str(qp), 
            '-preset', preset,
            output_filename) # output encoding
        
        if write_stderr_to_screen:
            self.ffmpeg_proc = subprocess.Popen(cmdstring, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE)
        else:
            self.ffmpeg_proc = subprocess.Popen(cmdstring, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=open('/dev/null', 'w'))       
    
    def write(self, frame):
        """Write a frame to the ffmpeg process"""
        self.ffmpeg_proc.stdin.write(frame.tostring())
    
    def write_bytes(self, bytestring):
        self.ffmpeg_proc.stdin.write(bytestring)
    
    def close(self):
        """Closes the ffmpeg process and returns stdout, stderr"""
        return self.ffmpeg_proc.communicate()


## Functions for creating output videos with overlays
def frame_update(
    ax, nframe, frame, plot_handles, node_positions, edge_names, im2, d_spatial, 
    d_temporal, node_plot_kwargs=None, edge_plot_kwargs=None,
    ):
    """Helper function to plot each frame.
    
    ax : the axis to plot in
    
    nframe : number of frame
        This is used to determine which whiskers and which contacts to plot
    
    frame : the image data
    
    plot_handles : list
        This contains handles to existing plots (lines, markers, etc) in the 
        current axis. First, all of these handles will be removed from the
        plot and from this list. Then, new handles will be created for data
        from this frame, plotted into the axis, and stored in this list.
    
    node_positions : DataFrame or None
        If not None, this contains the coordinates of named nodes.
        index : frame
        columns : (node_name, coord)
            'coord' has two values: 'x' and 'y'
    
    edge_names : list-like of 2-tuple-like
        A list of edges
        Each item is a pair of node names to be connected with an edge
    
    im2
    
    d_spatial
    
    d_temporal
    
    node_plot_kwargs : dict or None
        How to plot the nodes
    
    edge_plot_kwargs : dict
        How to plot the edges
    
    Returns: plot_handles
        These are returned so that they can be deleted next time
    """
    # set defaults
    if node_plot_kwargs is None:
        node_plot_kwargs = {
            'lw': 0,
            'marker': 'o',
            'ms': 6,
            'mfc': 'none',
            'color': 'yellow',
        }
    
    if edge_plot_kwargs is None:
        edge_plot_kwargs = {
            'marker': None,
            'lw': 1,
            'color': 'yellow',
            }
    
    # Get the frame
    im2.set_data(frame[::d_spatial, ::d_spatial])
    
    # Get the whiskers for this frame
    if node_positions is not None:
        # Remove old whiskers
        for handle in plot_handles:
            handle.remove()
        plot_handles = []            
        
        # Select out whiskers from this frame
        try:
            # This will become node_name on index and coord on columns
            frame_node_positions = node_positions.loc[nframe].unstack('coord')
        except KeyError:
            frame_node_positions = None
        
        if frame_node_positions is not None:
            # Plot all nodes
            handle, = ax.plot(
                frame_node_positions['x'].values,
                frame_node_positions['y'].values,
                **node_plot_kwargs,
                )
            
            # Store the handle to the nodes
            plot_handles.append(handle)

            # Plot edges
            # This ends up taking a bunch of time, try to optimize this
            if edge_names is not None:
                for edge_name in edge_names:
                    # Plot this edge
                    handle, = ax.plot(
                        frame_node_positions['x'].loc[edge_name],
                        frame_node_positions['y'].loc[edge_name],
                        **edge_plot_kwargs,
                        )
                    
                    # Store this edge
                    plot_handles.append(handle)
    
    return plot_handles

def write_video_with_overlays_from_data(output_filename, 
    input_reader, input_width, input_height,
    verbose=True,
    frame_triggers=None, trigger_dstart=-250, trigger_dstop=50,
    plot_trial_numbers=True,
    d_temporal=5, d_spatial=1,
    dpi=50, output_fps=30,
    input_video_alpha=1,
    node_positions=None,
    edge_names=None,
    write_stderr_to_screen=True,
    input_frame_offset=0,
    get_extra_text=None,
    text_size=10,
    ffmpeg_writer_kwargs=None,
    f=None, ax=None,
    func_update_figure=None,
    ):
    """Creating a video overlaid with whiskers, contacts, etc.
    
    The overall dataflow is this:
    1. Load chunks of frames from the input
    2. One by one, plot the frame with matplotlib. Overlay whiskers, edges,
        contacts, whatever.
    3. Dump the frame to an ffmpeg writer.
    
    # Input and output
    output_filename : file to create
    input_reader : PFReader or input video
    
    # Timing and spatial parameters
    frame_triggers : Only plot frames within (trigger_dstart, trigger_dstop)
        of a value in this array.
    trigger_dstart, trigger_dstop : number of frames
    d_temporal : Save time by plotting every Nth frame
    d_spatial : Save time by spatially undersampling the image
        The bottleneck is typically plotting the raw image in matplotlib
    
    # Video parameters
    dpi : The output video will always be pixel by pixel the same as the
        input (keeping d_spatial in mind). But this dpi value affects font
        and marker size.
    output_fps : set the frame rate of the output video (ffmpeg -r)
    input_video_alpha : alpha of image
    input_frame_offset : If you already seeked this many frames in the
        input_reader. Thus, now we know that the first frame to be read is
        actually frame `input_frame_offset` in the source (and thus, in
        the edge_a, contacts_table, etc.). This is the only parameter you
        need to adjust in this case, not frame_triggers or anything else.
    ffmpeg_writer_kwargs : other parameters for FFmpegWriter
    
    # Other sources of input
    edge_alpha : alpha of edge
    post_contact_linger : How long to leave the contact displayed    
        This is the total duration, so 0 will display nothing, and 1 is minimal.
    
    # Misc
    get_extra_text : if not None, should be a function that accepts a frame
        number and returns some text to add to the display. This is a 
        "real" frame number after accounting for any offset.
    text_size : size of the text
    contact_colors : list of color specs to use
    func_update_figure : optional, function that takes the frame number
        as input and updates the figure
    """
    # Parse the arguments
    frame_triggers = np.asarray(frame_triggers).astype(np.int)
    announced_frame_trigger = 0
    input_width = int(input_width)
    input_height = int(input_height)

    if ffmpeg_writer_kwargs is None:
        ffmpeg_writer_kwargs = {}


    ## Set up the graphical handles
    if verbose:
        print("setting up handles")

    if ax is None:
        # Create a figure with an image that fills it
        # We want the figsize to be in inches, so divide by dpi
        # And we want one invisible axis containing an image that fills the whole figure
        figsize = (input_width / float(dpi), input_height / float(dpi))
        f = plt.figure(frameon=False, dpi=(dpi / d_spatial), figsize=figsize)
        ax = f.add_axes([0, 0, 1, 1])
        ax.axis('off')
    
        # This return results in pixels, so should be the same as input width
        # and height. If not, probably rounding error above
        canvas_width, canvas_height = f.canvas.get_width_height()
        if (
            (input_width / d_spatial != canvas_width) or
            (input_height / d_spatial != canvas_height)
            ):
            raise ValueError("canvas size is not the same as input size")        
    else:
        assert f is not None
        
        # This is used later in creating the writer
        canvas_width, canvas_height = f.canvas.get_width_height()

    # Plot input video frames
    in_image = np.zeros((input_height, input_width))
    im2 = my.plot.imshow(in_image[::d_spatial, ::d_spatial], ax=ax, 
        axis_call='image', cmap=plt.cm.gray, 
        extent=(0, input_width, input_height, 0))
    im2.set_alpha(input_video_alpha)
    im2.set_clim((0, 255))

    # Text of trial
    if plot_trial_numbers:
        # Generate a handle to text
        txt = ax.text(
            .02, .02, 'waiting for text data',
            transform=ax.transAxes, # relative to axis size
            size=text_size, ha='left', va='bottom', color='w', 
            )
    
    # This will hold whisker objects
    whisker_handles = []
    
    # Create the writer
    writer = FFmpegWriter(
        output_filename=output_filename,
        frame_width=canvas_width,
        frame_height=canvas_height,
        output_fps=output_fps,
        input_pix_fmt='argb',
        write_stderr_to_screen=write_stderr_to_screen,
        **ffmpeg_writer_kwargs
        )
    
    ## Loop until input frames exhausted
    for nnframe, frame in enumerate(input_reader.iter_frames()):
        # Account for the fact that we skipped the first input_frame_offset frames
        nframe = nnframe + input_frame_offset
        
        # Break if we're past the last trigger
        if nframe > np.max(frame_triggers) + trigger_dstop:
            break
        
        # Skip if we're not on a dframe
        if np.mod(nframe, d_temporal) != 0:
            continue
        
        # Skip if we're not near a trial
        nearest_choice_idx = np.nanargmin(np.abs(frame_triggers - nframe))
        nearest_choice = frame_triggers[nearest_choice_idx]
        if not (nframe > nearest_choice + trigger_dstart and 
            nframe < nearest_choice + trigger_dstop):
            continue

        # Announce
        if ((announced_frame_trigger < len(frame_triggers)) and 
            (nframe > frame_triggers[announced_frame_trigger] + trigger_dstart)):
            print("Reached trigger for frame", frame_triggers[announced_frame_trigger])
            announced_frame_trigger += 1

        # Update the trial text
        if plot_trial_numbers:
            if get_extra_text is not None:
                extra_text = get_extra_text(nframe)
            else:
                extra_text = ''
            txt.set_text('frame %d %s' % (nframe, extra_text))

        # Update the frame
        whisker_handles = frame_update(
            ax, nframe, frame, whisker_handles, node_positions, edge_names,
            im2, 
            d_spatial, d_temporal,
            )
        
        if func_update_figure is not None:
            func_update_figure(nframe)
        
        # Write to pipe
        f.canvas.draw()
        string_bytes = f.canvas.tostring_argb()
        writer.write_bytes(string_bytes)
    
    ## Clean up
    if not input_reader.isclosed():
        input_reader.close()
    writer.close()
    plt.close(f)    