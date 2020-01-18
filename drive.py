import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.models import load_model
import h5py
from keras import __version__ as keras_version

from normalization import Normalization

import os

from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import tensorflow as tf

from openvino.inference_engine import IENetwork, IECore

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
exec_net = None
input_shape = None

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
def load_to_IE(model_xml):
    global exec_net, input_shape
    ### Load the Inference Engine API
    plugin = IECore()

    ### Load IR files into their related class
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    net = IENetwork(model=model_xml, weights=model_bin)

    ### Add a CPU extension, if applicable.
    plugin.add_extension(CPU_EXTENSION, "CPU")

    ### Get the supported layers of the network
    supported_layers = plugin.query_network(network=net, device_name="HETERO:MYRIAD,CPU")
    ### Check for any unsupported layers, and let the user
    ### know if anything is missing. Exit the program, if so.
    unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    if len(unsupported_layers) != 0:
        print("Unsupported layers found: {}".format(unsupported_layers))
        print("Check whether extensions are available to add to IECore.")
        return

    ### Load the network into the Inference Engine
    try:
        exec_net = plugin.load_network(net, "HETERO:MYRIAD,CPU")
        print("Loading to VPU/CPU...")
    except:
        exec_net = plugin.load_network(net, "CPU")
        print("Loading to CPU...")

    print("IR successfully loaded into Inference Engine.")

    input_blob = next(iter(net.inputs))

    input_shape = net.inputs[input_blob].shape

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        if exec_net is None:
            steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
        else:
            image_array = image_array[55:-25,0:320]            
            norm = Normalization()
            image_array = norm.call(image_array)
            image_array = image_array.transpose((2,0,1))
            image_array = image_array.reshape(1, 3, input_shape[2], input_shape[3])
            input_blob = next(iter(exec_net.inputs))
            it = iter(exec_net.outputs)
            output_blob = next(it)
            for output_blob in it:
                pass
        
            res = exec_net.infer({input_blob: image_array})
            steering_angle = res[output_blob][0][0]

        throttle = controller.update(float(speed))

        print(steering_angle, throttle)
        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    if (args.model.endswith('.h5')):
        # check that model Keras version is same as local Keras version
        f = h5py.File(args.model, mode='r')
        model_version = f.attrs.get('keras_version')
        keras_version = str(keras_version).encode('utf8')

        if model_version != keras_version:
            print('You are using Keras version ', keras_version,
                  ', but the model was built using ', model_version)

        model = load_model(args.model, custom_objects={'Normalization': Normalization()})
    else:
        load_to_IE(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

def export_keras_to_tf(input_model='model.h5', output_model='model.pb', num_output=1):
    print('Loading Keras model: ', input_model)

    keras_model = load_model(input_model, custom_objects={'Normalization': Normalization()})

    print(keras_model.summary())

    predictions = [None] * num_output
    prediction_node_names = [None] * num_output

    for i in range(num_output):
        prediction_node_names[i] = 'output_node' + str(i)
        predictions[i] = tf.identity(keras_model.outputs[i], 
        name=prediction_node_names[i])

    session = K.get_session()

    constant_graph = graph_util.convert_variables_to_constants(session, 
    session.graph.as_graph_def(), prediction_node_names)
    infer_graph = graph_util.remove_training_nodes(constant_graph) 

    graph_io.write_graph(infer_graph, '.', output_model, as_text=False)