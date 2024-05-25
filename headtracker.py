import cv2
import mediapipe as mp
import math
import vgamepad as vg

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

gamepad = vg.VX360Gamepad()
# configure the gamepad
Y_MULTIPLIER = 10
X_MULTIPLIER = 10

TILT_V_MULTIPLIER = 2.5
TILT_H_MULTIPLIER = 4

CAMERA = 1


# IGNORE THE FOLLOWING CODE IF YOU DONT KNOIW WHAT IT DOES
ZERO_SET = False

LEFT_LANDMARK = 334
RIGHT_LANDMARK = 105
BOTTOM_LANDMARK = 0

UPPER_LIP_ZERO_X = 0
UPPER_LIP_ZERO_Y = 0
UPPER_LIP_ZERO_Z = 0

LEFT_EYE_ZERO_X = 0
LEFT_EYE_ZERO_Y = 0
LEFT_EYE_ZERO_Z = 0

RIGHT_EYE_ZERO_X = 0
RIGHT_EYE_ZERO_Y = 0
RIGHT_EYE_ZERO_Z = 0

TILT_ANGLE_ZERO = 0
YAW_ANGLE_ZERO = 0

X_SERIES = [0]
Y_SERIES = [0]
TILT_SERIES = [0]
YAW_SERIES = [0]

DAMPENING = 10


def calculate_tilt_angle(vector):
    vertical_vector = (0, 1)

    dot_product = vector[1] * vertical_vector[1]

    magnitude = math.sqrt(vector[0]**2 + vector[1]**2)

    angle = math.acos(dot_product / magnitude)

    angle = math.degrees(angle)

    return angle


def calculate_yaw_angle(vector):
    horizontal_vector = (1, 0)

    dot_product = vector[0] * horizontal_vector[0]

    magnitude = math.sqrt(vector[0]**2 + vector[1]**2)

    angle = math.acos(dot_product / magnitude)

    angle = math.degrees(angle)

    return angle


def axis_in_bounds(axis_value):
    if axis_value > 32767:
        return 32767
    elif axis_value < -32768:
        return -32768
    else:
        return axis_value


def dampen_series(series):
    buffer = sorted(series)[int(DAMPENING/4): int(3 * DAMPENING/4)]
    if len(buffer) == 0:
        return axis_in_bounds(int(sum(series)/len(series)))
    return axis_in_bounds(int(sum(buffer)/len(buffer)))


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(CAMERA)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            cv2.circle(image, (int(results.multi_face_landmarks[0].landmark[RIGHT_LANDMARK].x * image.shape[1]), int(
                results.multi_face_landmarks[0].landmark[RIGHT_LANDMARK].y * image.shape[0])), 5, (0, 0, 255), 2)

            cv2.circle(image, (int(results.multi_face_landmarks[0].landmark[LEFT_LANDMARK].x * image.shape[1]), int(
                results.multi_face_landmarks[0].landmark[LEFT_LANDMARK].y * image.shape[0])), 5, (0, 0, 255), 2)

            cv2.circle(image, (int(results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].x * image.shape[1]), int(
                results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].y * image.shape[0])), 5, (0, 0, 255), 2)
            # calculate the distance between landmark RIGHT_LANDMARK and 0
            right_eye_to_upper_lip_distance = math.sqrt((results.multi_face_landmarks[0].landmark[RIGHT_LANDMARK].x - results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].x) ** 2 + (results.multi_face_landmarks[0].landmark[RIGHT_LANDMARK].y -
                                                                                                                                                                                           results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].y) ** 2 + (results.multi_face_landmarks[0].landmark[RIGHT_LANDMARK].z - results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].z) ** 2)
            # calculate the distance between landmark LEFT_LANDMARK and 0
            left_eye_to_upper_lip_distance = math.sqrt((results.multi_face_landmarks[0].landmark[LEFT_LANDMARK].x - results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].x) ** 2 + (results.multi_face_landmarks[0].landmark[LEFT_LANDMARK].y -
                                                                                                                                                                                         results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].y) ** 2 + (results.multi_face_landmarks[0].landmark[LEFT_LANDMARK].z - results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].z) ** 2)
            # average the distance
            average_distance = (
                right_eye_to_upper_lip_distance + left_eye_to_upper_lip_distance) / 2

            # normalize the distance
            gamepad_axis_value_head_distance = int(
                32768 * average_distance) * 10
            if gamepad_axis_value_head_distance > 32767:
                gamepad_axis_value_head_distance = 32767
            elif gamepad_axis_value_head_distance < -32768:
                gamepad_axis_value_head_distance = -32768

            # delta x and y
            gamepad_h_head_axis = results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].x - \
                UPPER_LIP_ZERO_X
            gamepad_v_head_axis = results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].y - \
                UPPER_LIP_ZERO_Y

            # normalize the delta x and y

            gamepad_axis_value_x = int(
                -32767 * gamepad_h_head_axis) * Y_MULTIPLIER
            gamepad_axis_value_y = int(
                -32767 * gamepad_v_head_axis) * X_MULTIPLIER

            # calculate the vertical tilt
            upper_lip_y = results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].y
            upper_lip_z = results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].z

            left_eye_y_vector = results.multi_face_landmarks[0].landmark[LEFT_LANDMARK].y - upper_lip_y
            left_eye_z_vector = results.multi_face_landmarks[0].landmark[LEFT_LANDMARK].z - upper_lip_z

            right_eye_y_vector = results.multi_face_landmarks[
                0].landmark[RIGHT_LANDMARK].y - upper_lip_y
            right_eye_z_vector = results.multi_face_landmarks[
                0].landmark[RIGHT_LANDMARK].z - upper_lip_z

            left_tilt_angle = calculate_tilt_angle(
                (left_eye_y_vector, left_eye_z_vector))

            right_tilt_angle = calculate_tilt_angle(
                (right_eye_y_vector, right_eye_z_vector))

            average_tilt_angle = (
                left_tilt_angle + right_tilt_angle) / 2

            delta_tile_angle = average_tilt_angle - TILT_ANGLE_ZERO
            normalized_tilt_angle = delta_tile_angle / 90

            gamepad_axis_value_tilt = int(
                32767 * normalized_tilt_angle) * TILT_V_MULTIPLIER

            # calculate the horizontal tilt

            right_eye_x = results.multi_face_landmarks[0].landmark[RIGHT_LANDMARK].x
            right_eye_z = results.multi_face_landmarks[0].landmark[RIGHT_LANDMARK].z

            left_eye_z = results.multi_face_landmarks[0].landmark[LEFT_LANDMARK].z

            left_eye_x_vector = results.multi_face_landmarks[0].landmark[LEFT_LANDMARK].x - right_eye_x
            left_eye_z_vector = results.multi_face_landmarks[0].landmark[LEFT_LANDMARK].z - right_eye_z

            yaw_angle = calculate_yaw_angle(
                (left_eye_x_vector, left_eye_z_vector))

            delta_yaw_angle = yaw_angle - YAW_ANGLE_ZERO
            if left_eye_z < right_eye_z:
                delta_yaw_angle = -delta_yaw_angle
            normalized_yaw_angle = delta_yaw_angle / 90

            gamepad_axis_value_yaw = int(
                32767 * normalized_yaw_angle) * TILT_H_MULTIPLIER

            # set the zero point
            if not ZERO_SET:
                UPPER_LIP_ZERO_X = results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].x
                UPPER_LIP_ZERO_Y = results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].y
                UPPER_LIP_ZERO_Z = results.multi_face_landmarks[0].landmark[BOTTOM_LANDMARK].z

                LEFT_EYE_ZERO_X = results.multi_face_landmarks[0].landmark[LEFT_LANDMARK].x
                LEFT_EYE_ZERO_Y = results.multi_face_landmarks[0].landmark[LEFT_LANDMARK].y
                LEFT_EYE_ZERO_Z = results.multi_face_landmarks[0].landmark[LEFT_LANDMARK].z

                RIGHT_EYE_ZERO_X = results.multi_face_landmarks[0].landmark[RIGHT_LANDMARK].x
                RIGHT_EYE_ZERO_Y = results.multi_face_landmarks[0].landmark[RIGHT_LANDMARK].y
                RIGHT_EYE_ZERO_Z = results.multi_face_landmarks[0].landmark[RIGHT_LANDMARK].z

                TILT_ANGLE_ZERO = delta_tile_angle
                YAW_ANGLE_ZERO = yaw_angle

                ZERO_SET = True

            X_SERIES.append(gamepad_axis_value_x)
            Y_SERIES.append(gamepad_axis_value_y)
            TILT_SERIES.append(gamepad_axis_value_tilt)
            YAW_SERIES.append(gamepad_axis_value_yaw)

            if len(X_SERIES) > DAMPENING:
                X_SERIES.pop(0)
                Y_SERIES.pop(0)
                TILT_SERIES.pop(0)
                YAW_SERIES.pop(0)

            gamepad.left_joystick(x_value=dampen_series(X_SERIES),
                                  y_value=dampen_series(Y_SERIES))

            gamepad.right_joystick(x_value=dampen_series(YAW_SERIES),
                                   y_value=dampen_series(TILT_SERIES))

            gamepad.update()
cap.release()
