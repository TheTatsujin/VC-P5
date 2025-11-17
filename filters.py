import cv2
from abc import ABC, abstractmethod

def put_overlay(img, x, y, w, h, alpha_mask, overlay_rgb):
    roi = img[y:y + h, x:x + w]

    for c in range(3):
        roi[:, :, c] = (alpha_mask * overlay_rgb[:, :, c] + (1 - alpha_mask) * roi[:, :, c])

    img[y:y + h, x:x + w] = roi

class Filter(ABC):
    @abstractmethod
    def apply(self, img, face_data):
        pass

class HappyFilter(Filter):
    def __init__(self, halo_image_path):
        self.halo_overlay = cv2.imread(halo_image_path, cv2.IMREAD_UNCHANGED)

    def apply(self, img, face_data):
        x, y, w, h = face_data['facial_area']['x'], face_data['facial_area']['y'], face_data['facial_area']['w'], face_data['facial_area']['h']
        offset_y = int(0.5 * h)
        halo_width = int(w * 1.5)
        halo_height = int(h * 0.3)

        halo_resized = cv2.resize(self.halo_overlay, (halo_width, halo_height))
        put_overlay(
            img,
            x=max(int(x - w / 4), 0),
            y=max(y - offset_y, 0),
            w=halo_width,
            h=halo_height,
            alpha_mask=halo_resized[:, :, 3] / 255.0,
            overlay_rgb=halo_resized[:, :, :3]
        )



class AngryFilter(Filter):
    def __init__(self, demon_img_path):
        self.demon_overlay = cv2.imread(demon_img_path, cv2.IMREAD_UNCHANGED)

    def apply(self, img, face_data):
        x, y, w, h = face_data['facial_area']['x'], face_data['facial_area']['y'], face_data['facial_area']['w'], face_data['facial_area']['h']
        offset_y = int(h*0.2)
        horns_width = w
        horns_height = int(h * 0.4)

        horns_resized = cv2.resize(self.demon_overlay, (horns_width, horns_height))
        put_overlay(
            img,
            x=x,
            y=max(y - offset_y, 0),
            w=horns_width,
            h=horns_height,
            alpha_mask=horns_resized[:, :, 3] / 255.0,
            overlay_rgb=horns_resized[:, :, :3]
        )

class SadFilter(Filter):
    def __init__(self, tears_img_path):
        self.tears_overlay = cv2.imread(tears_img_path, cv2.IMREAD_UNCHANGED)

    def apply(self, img, face_data):
        x, y, w, h = face_data['facial_area']['x'], face_data['facial_area']['y'], face_data['facial_area']['w'], face_data['facial_area']['h']
        left_eye_x, left_eye_y = face_data["facial_area"]["left_eye"]
        right_eye_x, right_eye_y = face_data["facial_area"]["right_eye"]

        offset_y = int(0.1 * h)
        tears_width = int(0.3 * w)
        tears_height = int(0.3 * h)
        tears_resized = cv2.resize(self.tears_overlay, (tears_width, tears_height))
        put_overlay(
            img,
            x=max(left_eye_x, 0),
            y=max(left_eye_y + offset_y, 0),
            w=tears_width,
            h=tears_height,
            alpha_mask=tears_resized[:, :, 3] / 255.0,
            overlay_rgb=tears_resized[:, :, :3]
        )
        tears_resized_flipped = cv2.flip(tears_resized, 1)
        offset_x = int(0.25 * w)
        put_overlay(
            img,
            x=max(right_eye_x - offset_x, 0),
            y=max(right_eye_y + offset_y, 0),
            w=tears_width,
            h=tears_height,
            alpha_mask=tears_resized_flipped[:, :, 3] / 255.0,
            overlay_rgb=tears_resized_flipped[:, :, :3]
        )

class FearFilter(Filter):
    def __init__(self, sweat_img_path):
        self.cold_sweat = cv2.imread(sweat_img_path, cv2.IMREAD_UNCHANGED)
    def apply(self, img, face_data):
        x, y, w, h = face_data['facial_area']['x'], face_data['facial_area']['y'], face_data['facial_area']['w'], face_data['facial_area']['h']

        img[y:y + int(h*0.5), x:x + w, 0] = img[y:y + int(h*0.5), x:x + w, 0] * 1.9

        cold_sweat_width = w
        cold_sweat_height = int(0.5 * h)
        cold_sweat_resized = cv2.resize(self.cold_sweat, (cold_sweat_width, cold_sweat_height))
        put_overlay(
            img,
            x=x,
            y=y,
            w=cold_sweat_width,
            h=cold_sweat_height,
            alpha_mask=cold_sweat_resized[:, :, 3] / 255.0,
            overlay_rgb=cold_sweat_resized[:, :, :3]
        )

class SurpriseFilter(Filter):
    def __init__(self, surprise_lines_img_path, surprised_eye_img_path):
        self.surprise_lines = cv2.imread(surprise_lines_img_path, cv2.IMREAD_UNCHANGED)
        self.surprised_eye = cv2.imread(surprised_eye_img_path, cv2.IMREAD_UNCHANGED)

    def _apply_surprise_lines(self, img, x, y, w, h):
        offset_y = int(0.5 * h)
        surprise_lines_width = int(w * 1.5)
        surprise_lines_height = int(h * 0.3)

        surprise_lines_resized = cv2.resize(self.surprise_lines, (surprise_lines_width, surprise_lines_height))
        put_overlay(
            img,
            x=max(int(x - w / 4), 0),
            y=max(y - offset_y, 0),
            w=surprise_lines_width,
            h=surprise_lines_height,
            alpha_mask=surprise_lines_resized[:, :, 3] / 255.0,
            overlay_rgb=surprise_lines_resized[:, :, :3]
        )
    def _apply_surprised_eyes(self, img, left_eye_x, left_eye_y, right_eye_x, right_eye_y, w, h):
        eye_offset_y = int(0.1 * h)
        surprised_eye_width = int(0.5 * w)
        surprised_eye_height = int(0.3 * h)
        surprised_eye_resized = cv2.resize(self.surprised_eye, (surprised_eye_width, surprised_eye_height))
        put_overlay(
            img,
            x=max(left_eye_x - int(w*0.18), 0),
            y=max(left_eye_y - eye_offset_y, 0),
            w=surprised_eye_width,
            h=surprised_eye_height,
            alpha_mask=surprised_eye_resized[:, :, 3] / 255.0,
            overlay_rgb=surprised_eye_resized[:, :, :3]
        )
        surprised_eye_resized_flipped = cv2.flip(surprised_eye_resized, 1)
        put_overlay(
            img,
            x=max(right_eye_x - int(w*0.35), 0),
            y=max(right_eye_y - eye_offset_y, 0),
            w=surprised_eye_width,
            h=surprised_eye_height,
            alpha_mask=surprised_eye_resized_flipped[:, :, 3] / 255.0,
            overlay_rgb=surprised_eye_resized_flipped[:, :, :3]
        )


    def apply(self, img, face_data):
        x, y, w, h = face_data['facial_area']['x'], face_data['facial_area']['y'], face_data['facial_area']['w'], face_data['facial_area']['h']
        left_eye_x, left_eye_y = face_data["facial_area"]["left_eye"]
        right_eye_x, right_eye_y = face_data["facial_area"]["right_eye"]
        self._apply_surprise_lines(img, x, y, w, h)
        self._apply_surprised_eyes(img, left_eye_x, left_eye_y, right_eye_x, right_eye_y, w, h)

class GalaxyFilter(Filter):
    def apply(self, img, face_data):
        print(face_data["face"])

class MediapipeDebugFilter(Filter):
    def __init__(self):
        self.mouth_color = (0, 143, 17)
    def apply(self, img, face_data):
        h, w, _ = img.shape
        for index, feature in enumerate(face_data.landmark):
            cv2.rectangle(img, (int(feature.x * w), int(feature.y * h)), (int(feature.x * w) + 3, int(feature.y * h) + 3), self.mouth_color, -1)
            #cv2.putText(img, str(index), (int(feature.x * w), int(feature.y * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 2)

class NoFilter(Filter):
    def apply(self, img, face_data):
        pass
