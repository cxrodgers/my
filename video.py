"""Generating or processing video, often using ffmpeg"""
from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import str
from builtins import map
from builtins import input
from builtins import object
from past.utils import old_div
import numpy as np
import subprocess
import re
import datetime
import os
import ffmpeg

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
        use_frame_time = (old_div(frame_number, float(frame_rate))) - .001
        use_frame_time = old_div(np.floor(use_frame_time * 1000), 1000.)
    
    elif frame_time is not None:
        frame_rate = get_video_params(filename)[2]
        use_frame_time = frame_time - (old_div(1., (2 * frame_rate)))
    
    else:
        raise ValueError("must specify frame by time or number")
    
    use_frame_string = '%0.3f' % use_frame_time
    return use_frame_string

def get_frame(filename, frametime=None, frame_number=None, frame_string=None,
    pix_fmt='gray', bufsize=10**9, path_to_ffmpeg='ffmpeg', vsync='drop'):
    """Returns a single frame from a video as an array.
    
    This creates an ffmpeg process and extracts data from it with a pipe.

    filename : video filename
    frame_string : to pass to -ss
    frametime, frame_number:
        If frame_string is None, then these are passed to 
        ffmpeg_frame_string to generate a frame string.
        
    pix_fmt : the "output" format of ffmpeg.
        currently only gray and rgb24 are accepted, because I need to 
        know how to reshape the result.
    
    This syntax is used to seek with ffmpeg:
        ffmpeg -ss %frametime% -i %filename% -vframes 1 ...
    This is supposed to be relatively fast while still accurate.
    
    TODO: Get this to return multiple frames from the same instance
    
    Returns:
        frame, stdout, stderr
        frame : 2d array, of shape (height, width)
        stdout : typically blank
        stderr : ffmpeg's text output
    """
    v_width, v_height = get_video_aspect(filename)
    
    if pix_fmt == 'gray':
        bytes_per_pixel = 1
        reshape_size = (v_height, v_width)
    elif pix_fmt == 'rgb24':
        bytes_per_pixel = 3
        reshape_size = (v_height, v_width, 3)
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
        '-vframes', '1',       
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
        read_size = bytes_per_pixel * v_width * v_height
        raw_image = pipe.stdout.read(read_size)    
        if len(raw_image) < read_size:
            raise OutOfFrames        
        flattened_im = np.fromstring(raw_image, dtype='uint8')
        frame = flattened_im.reshape(reshape_size)    
    
    except OutOfFrames:
        print("warning: cannot get frame")
        frame = None
    
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
    
    return frame, stdout, stderr


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
                print("warning: ran out of frames")
                out_of_frames = True
                this_chunk = old_div(len(raw_image), read_size_per_frame)
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
        raise ValueError("did not read the correct number of frames")

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
    if not os.path.exists(video_filename):
        raise ValueError("%s does not exist" % video_filename)
    
    probe = ffmpeg.probe(video_filename)
    assert len(probe['streams']) == 1
    
    # Container duration
    container_duration = float(probe['format']['duration'])
    
    # Stream duration
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
    
    # Check they are the same
    # Maybe to almost equal here
    assert stream_duration == container_duration
    
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