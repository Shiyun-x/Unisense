CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
PREDICT_TOKEN_INDEX = -300
DEPTH_TOKEN_INDEX = -400
SPEECH_TOKEN_INDEX = -500
POINTCLOUD_TOKEN_INDEX = -600
IMU_TOKEN_INDEX = -700
TACTILE_TOKEN_INDEX = -800
THERMAL_TOKEN_INDEX = -900
RADAR_TOKEN_INDEX = -1000
SMELL_TOKEN_INDEX = -1100
MOTION_TOKEN_INDEX = -1300
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_DEPTH_TOKEN = "<depth>"
DEFAULT_SPEECH_TOKEN = "<speech>"
DEFAULT_POINTCLOUD_TOKEN = "<pointcloud>"
DEFAULT_IMU_TOKEN = "<imu>"
DEFAULT_TACTILE_TOKEN = "<tactile>"
DEFAULT_THERMAL_TOKEN = "<thermal>"
DEFAULT_RADAR_TOKEN = "<radar>"
DEFAULT_SMELL_TOKEN = "<smell>"
DEFAULT_MOTION_TOKEN = "<motion>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
DEFAULT_PREDICT_TOKEN = "<predict>"

DESCRIPT_PROMPT = [
    "Describe this image thoroughly.",
    "Provide a detailed description in this picture.",
    "Detail every aspect of what's in this picture.",
    "Explain this image with precision and detail.",
    "Give a comprehensive description of this visual.",
    "Elaborate on the specifics within this image.",
    "Offer a detailed account of this picture's contents.",
    "Describe in detail what this image portrays.",
    "Break down this image into detailed descriptions.",
    "Provide a thorough description of the elements in this image."]
