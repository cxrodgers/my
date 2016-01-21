"""Generating or processing video, often using ffmpeg"""
import numpy as np
import subprocess
import re
import datetime
import os

class OutOfFrames(BaseException):
    """Exception raised when more frames cannot be extracted from a video"""
    pass

def get_frame(filename, frametime=None, frame_number=None, frame_string=None,
    pix_fmt='gray', bufsize=10**9):
    """Returns a single frame from a video as an array.
    
    This creates an ffmpeg process and extracts data from it with a pipe.

    filename : video filename
    frametime, frame_number : which frame to get
        if you request time T, ffmpeg gives you frame N, where N is 
        ceil(time * frame_rate). So -.001 gives you the first frame, and
        .001 gives you the second frame. It's hard to predict what will
        happen with one ms of the exact frame time due to rounding errors.
        
        So, if frame_number is not None:
            This passes ((frame_number / frame_rate) - 1 ms) rounded down
            to the nearest millisecond. This should give accurate results
            as long as frame rate is not >500 fps or so.
        else if frame_number is not None:
            Then this subtracts half a frame time from the frame time you
            requested, rounds to the nearest millisecond, and passes to ffmpeg.
            This will tend to give an unbiased estimate of the closest frame.
        else if frame_string is not None:
            The string is passed directly to ffmpeg
        
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
    
    # Choose the frame time string
    if frame_number is not None:
        frame_rate = get_video_params(filename)[2]
        use_frame_time = (frame_number / float(frame_rate)) - .001
        use_frame_time = np.floor(use_frame_time * 1000) / 1000.
        use_frame_string = '%0.3f' % use_frame_time
    elif frametime is not None:
        frame_rate = get_video_params(filename)[2]
        use_frame_time = frametime - (1. / (2 * frame_rate))
        use_frame_string = '%0.3f' % use_frame_time
    else:
        if frame_string is None:
            raise ValueError("must specify frame by time, number, or string")
        use_frame_string = frame_string
    
    # Create the command
    command = ['ffmpeg', 
        '-ss', use_frame_string,
        '-i', filename,
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
        print "warning: cannot get frame"
        frame = None
    
    finally:
        # Restore stdout
        pipe.terminate()

        # Keep the leftover data and the error signal (ffmpeg output)
        stdout, stderr = pipe.communicate()    
    
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
        raise ValueError, "mplayer not supported"
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
        print syscall
    if not dry_run:
        #os.system(syscall)
        syscall_l = syscall.split(' ')
        syscall_result = subprocess.check_output(syscall_l, 
            stderr=subprocess.STDOUT)
        if very_verbose:
            print syscall_result

def process_chunks_of_video(filename, n_frames, func='mean', verbose=False,
    frame_chunk_sz=1000, bufsize=10**9,
    image_w=None, image_h=None, pix_fmt='gray',
    finalize='concatenate'):
    """Read frames from video, apply function, return result
    
    Uses a pipe to ffmpeg to load chunks of frame_chunk_sz frames, applies
    func, then stores just the result of func to save memory.
    
    If n_frames > # available, returns just the available frames with a
    warning.
    
    filename : file to read
    n_frames : number of frames to process
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
    
    TODO: 
    if n_frames is None, set to max or inf
    get video params using ffprobe
    """
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
    command = ['ffmpeg', 
        '-i', filename,
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
                print frames_read
            # Figure out how much to acquire
            if frames_read + frame_chunk_sz > n_frames:
                this_chunk = n_frames - frames_read
            else:
                this_chunk = frame_chunk_sz
            
            # Read this_chunk, or as much as we can
            raw_image = pipe.stdout.read(read_size_per_frame * this_chunk)
            
            # check if we ran out of frames
            if len(raw_image) < read_size_per_frame * this_chunk:
                print "warning: ran out of frames"
                out_of_frames = True
                this_chunk = len(raw_image) / read_size_per_frame
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
            chunk_res = map(func, video)
            
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

    # Stick chunks together
    if len(res_l) == 0:
        print "warning: no data found"
        res = np.array([])
    elif finalize == 'concatenate':
        res = np.concatenate(res_l)
    elif finalize == 'listcomp':
        res = np.array([item for sublist in res_l for item in sublist])
    elif finalize == 'list':
        res = res_l
    else:
        print "warning: unknown finalize %r" % finalize
        res = res_l
        
    return res

def get_video_aspect(video_filename):
    """Returns width, height of video using ffprobe"""
    # Video duration and hence start time
    proc = subprocess.Popen(['ffprobe', video_filename],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = proc.communicate()[0]

    # Check if ffprobe failed, probably on a bad file
    if 'Invalid data found when processing input' in res:
        raise ValueError("Invalid data found by ffprobe in %s" % video_filename)
    
    # Find the video stream
    width_height_l = []
    for line in res.split("\n"):
        # Skip lines that aren't stream info
        if not line.strip().startswith("Stream #"):
            continue
        
        # Check that this is a video stream
        comma_split = line.split(',')
        if " Video: " not in comma_split[0]:
            continue
        
        # The third group should contain the size and aspect ratio
        if len(comma_split) < 3:
            raise ValueError("malform video stream string:", line)
        
        # The third group should contain the size and aspect, separated
        # by spaces
        size_and_aspect = comma_split[2].split()        
        if len(size_and_aspect) == 0:
            raise ValueError("malformed size/aspect:", comma_split[2])
        size_string = size_and_aspect[0]
        
        # The size should be two numbers separated by x
        width_height = size_string.split('x')
        if len(width_height) != 2:
            raise ValueError("malformed size string:", size_string)
        
        # Cast to int
        width_height_l.append(map(int, width_height))
    
    if len(width_height_l) > 1:
        print "warning: multiple video streams found, returning first"
    return width_height_l[0]


def get_video_duration(video_filename, return_as_timedelta=False):
    """Return duration of video using ffprobe"""
    # Video duration and hence start time
    proc = subprocess.Popen(['ffprobe', video_filename],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = proc.communicate()[0]

    # Check if ffprobe failed, probably on a bad file
    if 'Invalid data found when processing input' in res:
        raise ValueError(
            "Invalid data found by ffprobe in %s" % video_filename)

    # Parse out start time
    duration_match = re.search("Duration: (\S+),", res)
    assert duration_match is not None and len(duration_match.groups()) == 1
    video_duration_temp = datetime.datetime.strptime(
        duration_match.groups()[0], '%H:%M:%S.%f')
    video_duration = datetime.timedelta(
        hours=video_duration_temp.hour, 
        minutes=video_duration_temp.minute, 
        seconds=video_duration_temp.second,
        microseconds=video_duration_temp.microsecond)    
    
    if return_as_timedelta:
        return video_duration
    else:
        return video_duration.total_seconds()

def choose_rectangular_ROI(vfile, n_frames=4, interactive=False, check=True):
    """Displays a subset of frames from video so the user can specify an ROI.
    
    If interactive is False, the frames are simply displayed in a figure.
    If interactive is True, a simple text-based UI allows the user to input
    the x- and y- coordinates of the ROI. These are drawn and the user has
    the opportunity to confirm them.
    
    If check is True, then the values are swapped as necessary such that
    x0 < x1 and y0 < y1.
    
    Finally the results are returned as a dict with keys x0, x1, y0, y1.
    """
    import matplotlib.pyplot as plt
    import my.plot
    # Not sure why this doesn't work if it's lower down in the function
    if interactive:
        plt.ion()        

    # Get frames
    duration = get_video_duration(vfile)
    frametimes = np.linspace(0, duration, n_frames)
    frames = []
    for frametime in frametimes:
        frame, stdout, stderr = get_frame(vfile, frametime)
        frames.append(frame)
    
    # Plot them
    f, axa = plt.subplots(1, 4, figsize=(15, 4))
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
                    while True:
                        try:
                            val = raw_input("Enter %s: " % param)
                            break
                        except ValueError:
                            print "invalid entry"
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
                choice = raw_input("Confirm [y/n/q]: ")
                if choice == 'q':
                    res = {}
                    print "cancelled"
                    break
                elif choice == 'y':
                    break
                else:
                    pass
        except KeyboardInterrupt:
            res = {}
            print "cancelled"
        finally:
            plt.ioff()
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
        print ' '.join(syscall_l)
    
    # I think when -t parameter is set, it raises CalledProcessError
    #~ syscall_result = subprocess.check_output(syscall_l, 
        #~ stderr=subprocess.STDOUT)
    #~ if very_verbose:
        #~ print syscall_result
    os.system(' '.join(syscall_l))

def split():
    # ffmpeg -i 150401_CR1_cropped.mp4 -f segment -vcodec copy -reset_timestamps 1 -map 0 -segment_time 1000 OUTPUT%d.mp4
    pass


def get_video_params(video_filename):
    """Returns width, height, frame_rate of video using ffprobe"""
    # Video duration and hence start time
    proc = subprocess.Popen(['ffprobe', video_filename],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    res = proc.communicate()[0]

    # Check if ffprobe failed, probably on a bad file
    if 'Invalid data found when processing input' in res:
        raise ValueError("Invalid data found by ffprobe in %s" % video_filename)
    
    # Find the video stream
    width_height_l = []
    frame_rate_l = []
    for line in res.split("\n"):
        # Skip lines that aren't stream info
        if not line.strip().startswith("Stream #"):
            continue
        
        # Check that this is a video stream
        comma_split = line.split(',')
        if " Video: " not in comma_split[0]:
            continue
        
        # The third group should contain the size and aspect ratio
        if len(comma_split) < 3:
            raise ValueError("malform video stream string:", line)
        
        # The third group should contain the size and aspect, separated
        # by spaces
        size_and_aspect = comma_split[2].split()        
        if len(size_and_aspect) == 0:
            raise ValueError("malformed size/aspect:", comma_split[2])
        size_string = size_and_aspect[0]
        
        # The size should be two numbers separated by x
        width_height = size_string.split('x')
        if len(width_height) != 2:
            raise ValueError("malformed size string:", size_string)
        
        # Cast to int
        width_height_l.append(map(int, width_height))
    
        # The fourth group in comma_split should be %f fps
        frame_rate_fps = comma_split[4].split()
        if frame_rate_fps[1] != 'fps':
            raise ValueError("malformed frame rate:", frame_rate_fps)
        frame_rate_l.append(float(frame_rate_fps[0]))
    
    if len(width_height_l) > 1:
        print "warning: multiple video streams found, returning first"
    return width_height_l[0][0], width_height_l[0][1], frame_rate_l[0]


class WebcamController:
    def __init__(self, device='/dev/video0', output_filename='/dev/null',
        width=320, height=240, framerate=30,
        window_title='webcam', 
        ):
        """Init a new webcam controller for a certain webcam."""
        # Store params
        self.device = device
        self.output_filename = output_filename
        self.width = width
        self.height = height
        self.framerate = framerate
        self.window_title = window_title
        
        # Image controls
        self.image_controls = {
            'gain': 2,
            'exposure': 8,
            'brightness': 13,
            'contrast': 25,
            'saturation': 69,
            'hue': 0,
            'white_balance_automatic': 0,
            'gain_automatic': 0,
            'auto_exposure': 1, # flipped
            }
        
        self.read_stderr = None
        self.ffplay_stderr = None
        self.ffplay_stdout = None
    
    def start(self):
        """Start displaying and encoding
        
        To stop, call the stop method, or close the ffplay window.
        In the latter case, it will keep reading from the webcam until
        you call cleanup or delete the object.
        """
        # Set the image controls
        self.set_controls()
        
        #~ self.ffmpeg_proc = subprocess.Popen(['ffmpeg',
            #~ '-f', 'video4linux2',
            #~ '-i', self.device,
            #~ '-vcodec', 'mpeg4',
            #~ '-q', '2',
            #~ '-f', 'rawvideo', '-',
            
        
        # Create a process to read from the webcam
        self.read_proc = subprocess.Popen(['ffmpeg',
            '-f', 'video4linux2',
            '-i', self.device,
            '-vcodec', 'mpeg4',
            '-q', '2',
            '-f', 'rawvideo', '-',
            ], stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'))

        # Tee the compressed output to a file
        self.tee_proc = subprocess.Popen(['tee', self.output_filename], 
            stdin=self.read_proc.stdout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Play the output
        self.ffplay_proc = subprocess.Popen([
            'ffplay', 
            '-fflags', 'nobuffer',
            '-window_title', self.window_title,
            '-',
            ], 
            stdin=self.tee_proc.stdout,
            stdout=subprocess.PIPE, stderr=open(os.devnull, 'w'))

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
        for k, v in self.image_controls.items():
            cmd_list += ['-c', '%s=%d' % (k, v)]

        # Create a process to set the parameters and run it
        self.set_proc = subprocess.Popen(cmd_list,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.set_stdout, self.set_stderr = self.set_proc.communicate()

        if self.set_proc.returncode != 0:
            print "failed to set parameters"
            print self.set_stdout
            print self.set_stderr
            raise IOError("failed to set parameters")
    
    def stop(self):
        self.ffplay_proc.terminate()
        self.cleanup()
    
    def update(self):
        pass
    
    def cleanup(self):
        self.__del__()
    
    def __del__(self):
        if self.ffplay_proc.returncode is None:
            self.ffplay_stdout, self.ffplay_stderr = self.ffplay_proc.communicate()
        if self.read_proc.returncode is None:
            self.read_proc.terminate()
            self.read_proc.wait()
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
        print "update"
        data = self.ffplay_proc.stderr.read(1000000)
        print "got data"
        print len(data)
        while len(data) == 1000000:
            self.stderr_l.append(data)
            data = self.ffplay_proc.stderr.read(1000000)
        print "done"
    
    def __del__(self):
        try:
            if self.ffplay_proc.returncode is None:
                self.ffplay_stdout, self.ffplay_stderr = self.ffplay_proc.communicate()        
        except AttributeError:
            pass