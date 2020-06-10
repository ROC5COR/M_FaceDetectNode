import argparse
import os
import pickle
from wsgiref import simple_server
import json
import urllib

import cv2
import falcon
import imutils
import numpy as np
import threading
import base64

NODE_IDENTITY = "FaceDetectNode"

class FaceDetect:
    def __init__(self, model_folder):
        protoPath = os.path.sep.join([model_folder, "deploy.prototxt"])
        modelPath = os.path.sep.join([model_folder, "res10_300x300_ssd_iter_140000.caffemodel"])
        print("[INFO] loading face detector...")
        try:
            self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        except Exception as e:
            print("[ERROR] Error while loading OpenCV model for face detection: ", str(e))
            self.detector = None
        print("[INFO] Model loaded successfully")

    def detect_faces(self, decoded_image, confidence=0.5, only_roi_position=False, output='rgb'):
        if self.detector is None:
            print("[ERROR] Can't detect faces because detector is None")
            return
        image = imutils.resize(decoded_image, width=600)
        (h, w) = image.shape[:2]
        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        rois = []
        for i in range(0, detections.shape[2]):
            detection_confidence = detections[0, 0, i, 2]

            if detection_confidence > confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face position
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # Passing if face is too small
                if fW < 20 or fH < 20:
                    continue
                print("[FaceDetectNode:DEBUG] Face detected")
                if only_roi_position:
                    rois.append([int(startX), int(startY), int(endX), int(endY)])
                else:
                    if output=='rgb':
                        rois.append({'type':'image/rgb', 'data':face.tolist()})
                    elif output=='base64':
                        _ ,encoded = cv2.imencode('.jpg', face)
                        rois.append({
                            'type':'image/jpg', 'encoding':'base64', 
                            'data':base64.b64encode(encoded.tobytes()).decode('utf8')
                            })
                    else:
                        print("[FaceDetect:ERROR] Can't output data format of type: ",output)
                        rois.append(None)
                
        return rois


class FaceDetectNode(threading.Thread):
    def __init__(self, port):
        threading.Thread.__init__(self)

        current_file_dirname = os.path.dirname(os.path.abspath(__file__))
        face_detect_model_folder = os.path.join(current_file_dirname, "face_detection_model")
        print("[INFO:FaceDetectNode] Loading model from dir {}".format(face_detect_model_folder))
        self.face_detect = FaceDetect(face_detect_model_folder)
        self.port = port

        class MainRoute:
            def on_get(self, req, res):
                res.body = "FaceDetectNode"

        class Process:
            def __init__(self, parent):
                self.parent = parent

            def on_post(self, req, res):
                params = req.params
                data_received = req.bounded_stream.read()
                #key_values = data_received.split(b'&')
                #parsed_data = dict([x.split(b'=') for x in key_values])
                if data_received and 'mode' in params:
                    #byte_data = urllib.parse.unquote_to_bytes(parsed_data[b'data'])
                    np_arr = np.frombuffer(data_received, np.uint8)
                    img_np = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
                    if img_np is None:
                        res.body = "ko: failed to decode image"
                    else:
                        output = 'rgb'
                        if 'output' in params:
                            output = params['output']
                        
                        if params['mode'] == 'image':
                            faces = self.parent.face_detect.detect_faces(img_np, only_roi_position=False, output=output)
                            res.content_type = 'application/json'
                            res.body = json.dumps(faces)
                        elif params['mode'] == 'position':
                            faces = self.parent.face_detect.detect_faces(img_np, only_roi_position=True, output=output)
                            res.content_type = 'application/json'
                            res.body = json.dumps(faces)
                        else:
                            res.content_type = 'plain/text'
                            res.body = "mode is not recognized"
                else:
                    res.content_type = 'plain/text'
                    res.body = 'img data must be as byte data and mode must be as params'

            def on_get(self, req, res):
                res.body = "You have to POST an image as data and mode (image/position) as params"
        

        api = falcon.API()
        api.add_route('/', MainRoute())
        api.add_route('/process', Process(self))
        #api.add_route('/process_roi', ProcessROI(self))
        self.server = simple_server.make_server('', port, app=api)

    
    def run(self):
        print("[INFO] Starting server")
        self.server.serve_forever()
