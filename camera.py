import cv2  # Import the OpenCV library
import numpy as np
import time
import sys
import threading
import framebuffer
from pkg_resources import parse_version
from constants import Constants

gstreamerCameraFound = False
piCameraFound = False
piCamera2Found = False
try:
    import picamera
    from picamera.array import PiRGBArray
    piCameraFound = True
    print("Camera.py: Using picamera module")
except ImportError:
    print("Camera.py: picamera module not available")

try:
    import picamera2
    piCamera2Found = True
    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")

if not piCameraFound and not piCamera2Found:
    print("Camera.py: Using OpenCV interface instead")


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False."""
    # TODO: Problematic that gstreamerCameraFound is first set after
    #  instantiation of a Camera object
    return piCameraFound or piCamera2Found or gstreamerCameraFound

# Black magic in order to handle differences in namespace names in OpenCV 2.4 and 3.0
OPCV3 = parse_version(cv2.__version__) >= parse_version('3')

def capPropId(prop):
    """returns OpenCV VideoCapture property id given, e.g., "FPS
       This is needed because of differences in the Python interface in OpenCV 2.4 and 3.0
    """
    return getattr(cv2 if OPCV3 else cv2.cv, ("" if OPCV3 else "CV_") + "CAP_PROP_" + prop)


def gstreamer_pipeline(capture_width=1280, capture_height=720, framerate=30):
    """Utility function for setting parameters for the gstreamer camera pipeline"""
    return (
        "libcamerasrc !"
        "videobox autocrop=true !"
        "video/x-raw, width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "videoconvert ! "
        "appsink"
        % (
            capture_width,
            capture_height,
            framerate,
        )
    )



class CaptureThread(threading.Thread):
    """Internal worker thread that captures frames from the camera"""
    
    def __init__(self, cam, framebuffer):
        threading.Thread.__init__(self)
        self.cam = cam
        self.framebuffer = framebuffer
        self.terminateThreadEvent = threading.Event()


    def run(self):
        while not self.terminateThreadEvent.is_set():
            if piCameraFound:
                # Use piCamera

                #self.cam.capture(self.rawCapture, format="bgr", use_video_port=True)
                #image = self.rawCapture.array
            
                # clear the stream in preparation for the next frame
                #self.rawCapture.truncate(0)
                
                if sys.version_info[0] > 2:
                    # Python 3.x
                    image = np.empty((self.cam.resolution[1], self.cam.resolution[0], 3), dtype=np.uint8)
                else:
                    # Python 2.x
                    image = np.empty((self.cam.resolution[1] * self.cam.resolution[0] * 3,), dtype=np.uint8)
                    
                self.cam.capture(image, format="bgr", use_video_port=True)
                
                if sys.version_info[0] < 3:
                    # Python 2.x
                    image = image.reshape((self.cam.resolution[1], self.cam.resolution[0], 3))


            elif piCamera2Found:
                # TODO: Add support for pycamera2
                image = self.cam.capture_array("main")

            else:  # Use OpenCV
                retval, image = self.cam.read()  # Read frame

                if not retval:  # Error
                    print("CaptureThread: Could not read next frame")
                    exit(-1)

            # Update framebuffer
            self.framebuffer.new_frame(image)
        
        
    def stop(self):
        """Terminate the worker thread"""
        self.terminateThreadEvent.set()


class Camera(object):
    """This class is responsible for doing the image processing. It detects known landmarks and 
    measures distances and orientations to these."""
    
    def __init__(self, camidx, robottype='arlo', useCaptureThread=False):
        """Constructor:
             camidx - index of camera
             robottype - specify which robot you are using in order to use the correct camera calibration. 
                         Supported types: arlo, frindo, scribbler, macbookpro"""

        print("robottype =", robottype)
        self.useCaptureThread = useCaptureThread

        # TODO: Use this in the different camera configurations
        self.FPS = 5 # Framerate

        # Set camera calibration info
        if robottype == 'arlo':
            #self.imageSize = (1280, 720)
            self.imageSize = (1640, 1232)
            #self.intrinsic_matrix = np.asarray([ 7.1305391967046853e+02, 0., 3.1172820723774367e+02, 0.,
            #       7.0564929862291285e+02, 2.5634470978315028e+02, 0., 0., 1. ], dtype = np.float64)
            #self.intrinsic_matrix = np.asarray([ 6.0727040957659040e+02, 0., 3.0757300398967601e+02, 0.,
            #       6.0768864690145904e+02, 2.8935674612358201e+02, 0., 0., 1. ], dtype = np.float64)
            self.intrinsic_matrix = np.asarray([1687.0, 0., self.imageSize[0] / 2.0, 0.,
                   1687.0, self.imageSize[1] / 2.0, 0., 0., 1.], dtype = np.float64)
            # self.intrinsic_matrix = Constants.PID.CAMERA_MATRIX
            self.intrinsic_matrix.shape = (3, 3)
            #self.distortion_coeffs = np.asarray([ 1.1911006165076067e-01, -1.0003366233413549e+00,
            #       1.9287903277399834e-02, -2.3728201444308114e-03, -2.8137265581326476e-01 ], dtype = np.float64)
            self.distortion_coeffs = np.asarray([0., 0., 2.0546093607192093e-02, -3.5538453075048249e-03, 0.], dtype = np.float64)
        elif robottype == 'frindo':
            self.imageSize = (640, 480)
            #self.intrinsic_matrix = np.asarray([ 7.1305391967046853e+02, 0., 3.1172820723774367e+02, 0.,
            #       7.0564929862291285e+02, 2.5634470978315028e+02, 0., 0., 1. ], dtype = np.float64)
            #self.intrinsic_matrix = np.asarray([ 6.0727040957659040e+02, 0., 3.0757300398967601e+02, 0.,
            #       6.0768864690145904e+02, 2.8935674612358201e+02, 0., 0., 1. ], dtype = np.float64)
            self.intrinsic_matrix = np.asarray([500, 0., 3.0757300398967601e+02, 0.,
                   500, 2.8935674612358201e+02, 0., 0., 1.], dtype = np.float64)
            self.intrinsic_matrix.shape = (3, 3)
            #self.distortion_coeffs = np.asarray([ 1.1911006165076067e-01, -1.0003366233413549e+00,
            #       1.9287903277399834e-02, -2.3728201444308114e-03, -2.8137265581326476e-01 ], dtype = np.float64)
            self.distortion_coeffs = np.asarray([0., 0., 2.0546093607192093e-02, -3.5538453075048249e-03, 0.], dtype = np.float64)
        elif robottype == 'scribbler':
            # Unknown calibration - just using the Frindo calibration
            self.imageSize = (640, 480)
            self.intrinsic_matrix = np.asarray([7.1305391967046853e+02, 0., 3.1172820723774367e+02, 0.,
                   7.0564929862291285e+02, 2.5634470978315028e+02, 0., 0., 1.], dtype = np.float64)
            self.intrinsic_matrix.shape = (3, 3)
            self.distortion_coeffs = np.asarray([1.1911006165076067e-01, -1.0003366233413549e+00,
                   1.9287903277399834e-02, -2.3728201444308114e-03, -2.8137265581326476e-01], dtype = np.float64)
        elif robottype == 'macbookpro':
            self.imageSize = (1280, 720)
            #self.imageSize = (1080, 720)
            #self.intrinsic_matrix = np.asarray([ 8.6955302212233869e+02, 0., 5.2076864848745902e+02, 0.,
            #       8.7317664932843684e+02, 4.0331768178896669e+02, 0., 0., 1. ], dtype = np.float64)
            self.intrinsic_matrix = np.asarray([9.4328095162920715e+02, 0., self.imageSize[0] / 2.0, 0.,
                   9.4946668595979077e+02, self.imageSize[1] / 2.0, 0., 0., 1.], dtype = np.float64)
            self.intrinsic_matrix.shape = (3, 3)
            #self.distortion_coeffs = np.asarray([ -1.2844325433988565e-01, -1.3646926538980573e+00,
            #       -5.7263071423202944e-03, 5.7422957803983802e-03, 5.9722099836744738e+00 ], dtype = np.float64)
            self.distortion_coeffs = np.asarray([0., 0., -1.6169374082976234e-02, 8.7657653170062459e-03, 0.], dtype = np.float64)
        else:
            print("Camera.__init__: Unknown robot type")
            exit(-1)
            

        # Open a camera device for capturing                 
        if piCameraFound:
            # piCamera is available so we use this
            #self.cam = picamera.PiCamera(camidx)
            self.cam = picamera.PiCamera(camera_num=camidx, resolution=self.imageSize, framerate=30)
            
            if not self.useCaptureThread:
                self.rawCapture = PiRGBArray(self.cam, size=self.cam.resolution)
                
            time.sleep(2) # wait for the camera            
            
            # Too slow code - instead just use cam.capture
            #self.capture_generator = self.cam.capture_continuous(self.rawCapture, format="bgr", use_video_port=True)

            gain = self.cam.awb_gains
            self.cam.awb_mode='off'
            self.cam.awb_gains = gain
            
            self.cam.shutter_speed = self.cam.exposure_speed
            self.cam.exposure_mode = 'off'


            print("shutter_speed = ", self.cam.shutter_speed)
            print("awb_gains = ", self.cam.awb_gains)
            
            
            print("Camera width = ", self.cam.resolution[0])
            print("Camera height = ", self.cam.resolution[1])
            print("Camera FPS = ", self.cam.framerate)

        elif piCamera2Found:
            # TODO: Add support for pycamera2
            self.cam = picamera2.Picamera2()
            #self.picam2_config = self.cam.create_still_configuration()
            frame_duration_limit = int(1/self.FPS * 1000000) # Microseconds
            # TODO: Change configuration to set resolution, framerate
            self.picam2_config = self.cam.create_video_configuration({"size": self.imageSize, "format": 'RGB888'},
                                                                     controls={"FrameDurationLimits": (frame_duration_limit, frame_duration_limit),
                                                                               "ScalerCrop": (0,0,3280,2464)},
                                                                     queue=False)
            self.cam.configure(self.picam2_config) # Not really necessary
            self.cam.start(show_preview=False)

            time.sleep(1)  # wait for camera to setup

        else:  # Use OpenCV interface

            # We next try the gstreamer interface
            self.cam = cv2.VideoCapture(0) # i changed this because VideoCapture with gstreamer fails but has isOpened
            # self.cam = cv2.VideoCapture(gstreamer_pipeline(
            #     capture_width=self.imageSize[0], capture_height=self.imageSize[1]),
            #     apiPreference=cv2.CAP_GSTREAMER)
            if not self.cam.isOpened(): # Did not work

                # We try first the generic auto-detect interface
                self.cam = cv2.VideoCapture(camidx)
                if not self.cam.isOpened():  # Error
                    print("Camera.__init__: Could not open camera")
                    exit(-1)
                else:
                    print("Camera.__init__: Using OpenCV with auto-detect interface")
            else:
                gstreamerCameraFound = True
                print("Camera.__init__: Using OpenCV with gstreamer")

            time.sleep(1) # wait for camera
            
            # Set camera properties
            self.cam.set(capPropId("FRAME_WIDTH"), self.imageSize[0])
            self.cam.set(capPropId("FRAME_HEIGHT"), self.imageSize[1])
            #self.cam.set(capPropId("BUFFERSIZE"), 1) # Does not work
            #self.cam.set(capPropId("FPS"), 15)
            self.cam.set(capPropId("FPS"), 30)
        
            time.sleep(1)
        
            # Get camera properties
            print("Camera width = ", int(self.cam.get(capPropId("FRAME_WIDTH"))))
            print("Camera height = ", int(self.cam.get(capPropId("FRAME_HEIGHT"))))
            print("Camera FPS = ", int(self.cam.get(capPropId("FPS"))))
        
        # Initializing the camera distortion maps
        #self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.intrinsic_matrix, self.distortion_coeffs, np.eye(3,3), np.eye(3,3), self.imageSize, cv2.CV_32FC1)
        # Not needed we use the cv2.undistort function instead
        
        # Initialize chessboard pattern parameters
        self.patternFound = False
        self.patternSize = (3,4)
        self.patternUnit = 50.0 # mm (size of one checker square)
        self.corners = []

        # Initialize aruco detector
        self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        # Set the correct physical marker size here
        self.arucoMarkerLength = 0.15  # [m] actual size of aruco markers (in object coordinate system)
        
        # Initialize worker thread and framebuffer
        if self.useCaptureThread:
            print("Using capture thread")
            self.framebuffer = framebuffer.FrameBuffer((self.imageSize[1], self.imageSize[0], 3))
            self.capturethread = CaptureThread(self.cam, self.framebuffer)
            self.capturethread.start()
            time.sleep(0.75)

    def __del__(self):

        # TODO: Add support for pycamera2
        if piCameraFound or piCamera2Found:
            self.cam.close()

        
    def terminateCaptureThread(self):
        if self.useCaptureThread:
            self.capturethread.stop()
            self.capturethread.join()
        
    def get_capture(self):
        """Access to the internal camera object for advanced control of the camera."""
        return self.cam

    def get_colour(self):
        """OBSOLETE - use instead get_next_frame"""
        print("OBSOLETE get_colour - use instead get_next_frame")
        return self.get_next_frame()

    def get_next_frame(self):
        """Gets the next available image frame from the camera."""
        if self.useCaptureThread:
            img = self.framebuffer.get_frame()
            
            if img is None:
                img = np.array((self.imageSize[0], self.imageSize[1], 3), dtype=np.uint8)
                        
        else:

            if piCameraFound:
                # Use picamera module
            
                self.cam.capture(self.rawCapture, format="bgr", use_video_port=True)
                img = self.rawCapture.array
            
                # clear the stream in preparation for the next frame
                self.rawCapture.truncate(0)

            elif piCamera2Found:
                # Use pycamera2 module
                img = self.cam.capture_array("main")

            else: # Use OpenCV
                retval, img = self.cam.read()  # Read frame

                if not retval:  # Error
                    print("Camera.get_colour: Could not read next frame")
                    exit(-1)
        
        return img
    # ArUco object detector
    def _detect_aruco_objects(self, img):
        """Detect objects in the form of a binary ArUco code and return object IDs, distances (in cm) and
        angles (in radians) to detected ArUco codes. The angle is computed as the signed angle between
        translation vector to detected object projected onto the x-z plabe and the z-axis (pointing out
        of the camera). This corresponds to that the angle is measuring location along the horizontal x-axis.

        If no object is detected, the returned variables are set to None."""
        self.aruco_corners, self.ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, self.arucoDict)
        self.rvecs, self.tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(self.aruco_corners, self.arucoMarkerLength, self.intrinsic_matrix, self.distortion_coeffs)


        if not isinstance(self.ids, type(None)):
            dists = np.linalg.norm(self.tvecs, axis=len(self.tvecs.shape) - 1) * 100
            # Make sure we always return properly shaped arrays
            dists = dists.reshape((dists.shape[0],))
            ids = self.ids.reshape((self.ids.shape[0],))

            # Compute angles
            angles = np.zeros(dists.shape, dtype=dists.dtype)
            for i in range(dists.shape[0]):
                tobj = self.tvecs[i] * 100 / dists[i]
                zaxis = np.zeros(tobj.shape, dtype=tobj.dtype)
                zaxis[0,-1] = 1
                xaxis = np.zeros(tobj.shape, dtype=tobj.dtype)
                xaxis[0,0] = 1

                # We want the horizontal angle so project tobjt onto the x-z plane
                tobj_xz = tobj
                tobj_xz[0,1] = 0
                # Should the sign be clockwise or counter-clockwise (left or right)?
                # In this version it is positive to the left as seen from the camera.
                direction = -1*np.sign(tobj_xz[0,0])  # The same as np.sign(np.dot(tobj, xaxis.T))
                angles[i] = direction * np.arccos(np.dot(tobj_xz, zaxis.T))
        else:
            dists = None
            ids = None
            angles = None
        return ids, dists, angles


    def detect_aruco_objects(self, img):
        ids, dists, angles = self._detect_aruco_objects()
        if ids is None:
            return (None, None, None)
        r = Constants.Robot.RADIUS*10
        Lx = r + np.cos(angles)
        Ly = distance * np.sin(angle)
        dists_new = np.sqrt(Lx**2 + Ly**2)
        angles_new = np.arcsin(Ly/dists_new)
        return ids, dists_new, angles_new


    def draw_aruco_objects(self, img):
        """Draws detected objects and their orientations on the image given in img."""
        if not isinstance(self.ids, type(None)):
            outimg = cv2.aruco.drawDetectedMarkers(img, self.aruco_corners, self.ids)
            for i in range(self.ids.shape[0]):
                outimg = cv2.drawFrameAxes(outimg, self.intrinsic_matrix, self.distortion_coeffs,
                                           self.rvecs[i], self.tvecs[i], self.arucoMarkerLength)
        else:
            outimg = img

        return outimg

    # Chessboard object detector
    def get_object(self, img):
        """Detect object and return object type, distance (in cm), angle (in radians) and 
        colour probability table in the order (R,G,B)"""
        objectType = 'none'
        colourProb = np.ones((3,)) / 3.0
        distance = 0.0
        angle = 0.0
        self.patternFound = False
        
        #patternFound, corners = self.get_corners(img)
        self.get_corners(img)
        
        if (self.patternFound):
            
            # Determine if the object is horizontal or vertical
            delta_x = abs(self.corners[0, 0, 0] - self.corners[2, 0, 0])
            delta_y = abs(self.corners[0, 0, 1] - self.corners[2, 0, 1])
            horizontal = (delta_y > delta_x)
            if (horizontal):
                objectType = 'horizontal'
            else:
                objectType = 'vertical'
            
            # Compute distances and angles
            if (horizontal):
                height = ((abs (self.corners[0, 0, 1] - self.corners[2, 0, 1]) +
                          abs (self.corners[9, 0, 1] - self.corners[11, 0, 1])) / 2.0)
                          
                patternHeight = (self.patternSize[0]-1.0) * self.patternUnit
                
            else:
                height = (abs (self.corners[0, 0, 1] - self.corners[9, 0, 1]) +
                              abs (self.corners[2, 0, 1] - self.corners[11, 0, 1])) / 2.0
                              
                patternHeight = (self.patternSize[1]-1.0) * self.patternUnit
            
            #distance = 1.0 / (0.0001 * height);
            distance = self.intrinsic_matrix[1, 1] * patternHeight / (height * 10.0)

            center = (self.corners[0, 0, 0] + self.corners[2, 0, 0] +
                                         self.corners[9, 0, 0] + self.corners[11, 0, 0]) / 4.0
            
            #angle = 0.0018 * center - 0.6425;
            #angle *= -1.0;
            angle = -np.arctan2(center - self.intrinsic_matrix[0, 2], self.intrinsic_matrix[0, 0])
            
            
            #### Classify object by colour
            
            # Extract rectangle corners
            points = np.array(
                        [
                         self.corners[0],
                         self.corners[2],
                         self.corners[9],
                         self.corners[11]
                        ]
                      )
                    
            points.shape = (4, 2)
            points = np.int32(points)

            # Compute region of interest
            mask = np.zeros((self.imageSize[1], self.imageSize[0]), dtype = np.uint8) 
            #cv2.fillConvexPoly (mask, points, (255, 255, 255))
            cv2.fillConvexPoly (mask, points, 255)    
    
            # Compute mean colour inside region of interest
            mean_colour = cv2.mean(img, mask)  # There is a bug here in Python 3 - it produces a segfault
            
            red = mean_colour[2]
            green = mean_colour[1]
            blue = mean_colour[0]
            sum = red + green + blue

            colourProb[0] = red / sum
            colourProb[1] = green / sum
            colourProb[2] = blue / sum
            
            
        return objectType, distance, angle, colourProb
        
        
    def get_corners(self, img):
        """Detect corners - this is an auxillary method and should not be used directly"""
        
        # Convert to gray scale
        gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
        loggray = cv2.log(gray + 1.0)
        cv2.normalize(loggray,loggray,0,255,cv2.NORM_MINMAX)
        gray = cv2.convertScaleAbs(loggray)
        
        #retval, self.corners = cv2.findChessboardCorners(gray, self.patternSize, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
        retval, self.corners = cv2.findChessboardCorners(gray, self.patternSize, cv2.CALIB_CB_FAST_CHECK)
        if (retval > 0):
            self.patternFound = True
            #cv2.cornerSubPix(gray, self.corners, (5,5), (-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 3, 0.0))
            
        return self.patternFound, self.corners
        
        
    def draw_object(self, img):
        """Draw the object if found into img"""
        cv2.drawChessboardCorners(img, self.patternSize, self.corners, self.patternFound)
        


if (__name__=='__main__'):
    print("Opening and initializing camera")
    
    #cam = Camera(0, 'macbookpro', useCaptureThread=True)
    #cam = Camera(0, 'macbookpro', useCaptureThread = False)
    #cam = Camera(0, robottype='arlo', useCaptureThread=True)
    cam = Camera(0, robottype='arlo', useCaptureThread=False)
    
    # Open a window
    WIN_RF1 = "Camera view"
    cv2.namedWindow(WIN_RF1)
    cv2.moveWindow(WIN_RF1, 50, 50)
        
    #WIN_RF3 = "Camera view - gray"
    #cv2.namedWindow(WIN_RF3)
    #cv2.moveWindow(WIN_RF3, 550, 50)
    
    while True:
        
        action = cv2.waitKey(10)
        
        if action == ord('q'):  # Quit
            break
    
        # Fetch next frame
        #colour = cam.get_colour()
        colour = cam.get_next_frame()
                
        # Convert to gray scale
        #gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY )
        #loggray = cv2.log(gray + 1.0)
        #cv2.normalize(loggray, loggray, 0, 255, cv2.NORM_MINMAX)
        #gray = cv2.convertScaleAbs(loggray)
        
        # Detect objects
        objectType, distance, angle, colourProb = cam.get_object(colour)
        if objectType != 'none':
           print("Object type = ", objectType, ", distance = ", distance, ", angle = ", angle, ", colourProb = ", colourProb)

        # Draw detected pattern
        #cam.draw_object(colour)

        IDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(IDs, type(None)):
            for i in range(len(IDs)):
                print("Object ID = ", IDs[i], ", Distance = ", dists[i], ", angles = ", angles[i])

        # Draw detected objects
        cam.draw_aruco_objects(colour)

        # Show frames
        cv2.imshow(WIN_RF1, colour)
        
        # Show frames
        #cv2.imshow(WIN_RF3, gray)
        
        
    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()
