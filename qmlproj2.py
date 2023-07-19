# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:34:08.354275Z","iopub.execute_input":"2023-07-19T13:34:08.355000Z","iopub.status.idle":"2023-07-19T13:34:50.690167Z","shell.execute_reply.started":"2023-07-19T13:34:08.354962Z","shell.execute_reply":"2023-07-19T13:34:50.689027Z"},"jupyter":{"outputs_hidden":false}}
!pip install qiskit

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:34:50.692743Z","iopub.execute_input":"2023-07-19T13:34:50.693365Z","iopub.status.idle":"2023-07-19T13:35:13.224933Z","shell.execute_reply.started":"2023-07-19T13:34:50.693325Z","shell.execute_reply":"2023-07-19T13:35:13.223580Z"},"jupyter":{"outputs_hidden":false}}
!pip install qiskit_machine_learning

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:35:13.230420Z","iopub.execute_input":"2023-07-19T13:35:13.231232Z","iopub.status.idle":"2023-07-19T13:35:26.614087Z","shell.execute_reply.started":"2023-07-19T13:35:13.231190Z","shell.execute_reply":"2023-07-19T13:35:26.612878Z"},"jupyter":{"outputs_hidden":false}}
!pip install qiskit tensorflow scikit-learn

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:35:26.617048Z","iopub.execute_input":"2023-07-19T13:35:26.617856Z","iopub.status.idle":"2023-07-19T13:35:26.954479Z","shell.execute_reply.started":"2023-07-19T13:35:26.617816Z","shell.execute_reply":"2023-07-19T13:35:26.953579Z"},"jupyter":{"outputs_hidden":false}}
import qiskit

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:35:26.955829Z","iopub.execute_input":"2023-07-19T13:35:26.956803Z","iopub.status.idle":"2023-07-19T13:35:37.275770Z","shell.execute_reply.started":"2023-07-19T13:35:26.956766Z","shell.execute_reply":"2023-07-19T13:35:37.274608Z"},"jupyter":{"outputs_hidden":false}}
import tensorflow as tf

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:35:37.277034Z","iopub.execute_input":"2023-07-19T13:35:37.277845Z","iopub.status.idle":"2023-07-19T13:35:37.702414Z","shell.execute_reply.started":"2023-07-19T13:35:37.277811Z","shell.execute_reply":"2023-07-19T13:35:37.701394Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np
from qiskit import QuantumCircuit, Aer
from qiskit.visualization import plot_histogram
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:35:37.703915Z","iopub.execute_input":"2023-07-19T13:35:37.704272Z","iopub.status.idle":"2023-07-19T13:35:38.103321Z","shell.execute_reply.started":"2023-07-19T13:35:37.704238Z","shell.execute_reply":"2023-07-19T13:35:38.102314Z"},"jupyter":{"outputs_hidden":false}}
# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:36:19.729317Z","iopub.execute_input":"2023-07-19T13:36:19.729854Z","iopub.status.idle":"2023-07-19T13:36:19.998492Z","shell.execute_reply.started":"2023-07-19T13:36:19.729816Z","shell.execute_reply":"2023-07-19T13:36:19.997373Z"},"jupyter":{"outputs_hidden":false}}
# Normalize the data to [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:38:35.924158Z","iopub.execute_input":"2023-07-19T13:38:35.924546Z","iopub.status.idle":"2023-07-19T13:38:35.930575Z","shell.execute_reply.started":"2023-07-19T13:38:35.924514Z","shell.execute_reply":"2023-07-19T13:38:35.929447Z"},"jupyter":{"outputs_hidden":false}}
from qiskit.circuit.library import ZZFeatureMap

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:45:06.300328Z","iopub.execute_input":"2023-07-19T13:45:06.300724Z","iopub.status.idle":"2023-07-19T13:45:06.305405Z","shell.execute_reply.started":"2023-07-19T13:45:06.300693Z","shell.execute_reply":"2023-07-19T13:45:06.304353Z"},"jupyter":{"outputs_hidden":false}}
from qiskit import Aer, QuantumCircuit, transpile, assemble

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:52:58.300324Z","iopub.execute_input":"2023-07-19T13:52:58.300790Z","iopub.status.idle":"2023-07-19T13:52:58.306767Z","shell.execute_reply.started":"2023-07-19T13:52:58.300756Z","shell.execute_reply":"2023-07-19T13:52:58.305573Z"},"jupyter":{"outputs_hidden":false}}

# Define the quantum feature map (circuit)
num_qubits = 4
num_rotations = 2

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:36:21.237939Z","iopub.execute_input":"2023-07-19T13:36:21.238283Z","iopub.status.idle":"2023-07-19T13:36:21.244826Z","shell.execute_reply.started":"2023-07-19T13:36:21.238253Z","shell.execute_reply":"2023-07-19T13:36:21.243423Z"},"jupyter":{"outputs_hidden":false}}
# # Add quantum gates to create the desired feature map
# # For example, let's use simple Pauli-X gates
# for i in range(num_qubits):
#     qc.h(i)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:53:12.019636Z","iopub.execute_input":"2023-07-19T13:53:12.020000Z","iopub.status.idle":"2023-07-19T13:53:12.025094Z","shell.execute_reply.started":"2023-07-19T13:53:12.019972Z","shell.execute_reply":"2023-07-19T13:53:12.023730Z"},"jupyter":{"outputs_hidden":false}}
from qiskit.quantum_info import Statevector

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:53:14.901834Z","iopub.execute_input":"2023-07-19T13:53:14.902204Z","iopub.status.idle":"2023-07-19T13:53:14.907113Z","shell.execute_reply.started":"2023-07-19T13:53:14.902174Z","shell.execute_reply":"2023-07-19T13:53:14.905984Z"},"jupyter":{"outputs_hidden":false}}
# Encode the classical images into quantum states using the quantum feature map
backend = Aer.get_backend('statevector_simulator')

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:46:57.038221Z","iopub.execute_input":"2023-07-19T13:46:57.038612Z","iopub.status.idle":"2023-07-19T13:46:57.054085Z","shell.execute_reply.started":"2023-07-19T13:46:57.038579Z","shell.execute_reply":"2023-07-19T13:46:57.052906Z"},"jupyter":{"outputs_hidden":false}}
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     print('GPU device not found. Using CPU instead.')
#     device_name = '/device:CPU:0'

# # Assign the process to a GPU
# with tf.device(device_name):
#     for layer in range(num_rotations):
#         for qubit in range(num_qubits):
#             qc.u(params[layer, qubit, 0], params[layer, qubit, 1], params[layer, qubit, 2], qubit)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:46:57.901779Z","iopub.execute_input":"2023-07-19T13:46:57.902574Z","iopub.status.idle":"2023-07-19T13:46:57.907982Z","shell.execute_reply.started":"2023-07-19T13:46:57.902535Z","shell.execute_reply":"2023-07-19T13:46:57.906650Z"},"jupyter":{"outputs_hidden":false}}
# # Encode the classical images into quantum states using the quantum feature map
# backend = Aer.get_backend('statevector_simulator')

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:57:16.024485Z","iopub.execute_input":"2023-07-19T13:57:16.024876Z","iopub.status.idle":"2023-07-19T13:57:16.032355Z","shell.execute_reply.started":"2023-07-19T13:57:16.024846Z","shell.execute_reply":"2023-07-19T13:57:16.031146Z"},"jupyter":{"outputs_hidden":false}}
def quantum_feature_map(image, params):
    # Create a new quantum circuit for each image and apply the parameterized rotations
    num_qubits = 4
    num_rotations = 2
    qc = QuantumCircuit(num_qubits)

    for layer in range(num_rotations):
        for qubit in range(num_qubits):
            qc.u(params[layer, qubit, 0], params[layer, qubit, 1], params[layer, qubit, 2], qubit)
    
    # Get the statevector after applying the parameterized rotations
    tqc = transpile(qc, backend)
    job = assemble(tqc)
    result = backend.run(job).result().get_statevector()
    return np.real(result)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:57:19.399182Z","iopub.execute_input":"2023-07-19T13:57:19.399561Z","iopub.status.idle":"2023-07-19T13:57:19.418536Z","shell.execute_reply.started":"2023-07-19T13:57:19.399530Z","shell.execute_reply":"2023-07-19T13:57:19.417623Z"},"jupyter":{"outputs_hidden":false}}
# Generate random initial parameters for each image
params = np.random.rand(len(X_train), num_rotations, num_qubits, 3) * 2 * np.pi

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:57:20.405300Z","iopub.execute_input":"2023-07-19T13:57:20.405736Z","iopub.status.idle":"2023-07-19T14:04:19.808090Z","shell.execute_reply.started":"2023-07-19T13:57:20.405704Z","shell.execute_reply":"2023-07-19T14:04:19.807076Z"},"jupyter":{"outputs_hidden":false}}
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU device not found. Using CPU instead.')
    device_name = '/device:CPU:0'

# Assign the process to a GPU
with tf.device(device_name):
    # Preprocess the data using the quantum feature map and store the results
    quantum_train_images = np.array([quantum_feature_map(image.flatten(), params[i]) for i, image in enumerate(X_train)])
    quantum_val_images = np.array([quantum_feature_map(image.flatten(), params[i]) for i, image in enumerate(X_val)])

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T14:05:12.733192Z","iopub.execute_input":"2023-07-19T14:05:12.733579Z","iopub.status.idle":"2023-07-19T14:05:12.765765Z","shell.execute_reply.started":"2023-07-19T14:05:12.733545Z","shell.execute_reply":"2023-07-19T14:05:12.764756Z"},"jupyter":{"outputs_hidden":false}}
# Define the classical neural network model using TensorFlow/Keras
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(2 ** num_qubits,)),
    layers.Dense(10, activation='softmax')
])

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T14:05:13.891984Z","iopub.execute_input":"2023-07-19T14:05:13.892332Z","iopub.status.idle":"2023-07-19T14:05:13.911916Z","shell.execute_reply.started":"2023-07-19T14:05:13.892302Z","shell.execute_reply":"2023-07-19T14:05:13.910876Z"},"jupyter":{"outputs_hidden":false}}
# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T14:05:16.440913Z","iopub.execute_input":"2023-07-19T14:05:16.441273Z","iopub.status.idle":"2023-07-19T14:06:39.035925Z","shell.execute_reply.started":"2023-07-19T14:05:16.441240Z","shell.execute_reply":"2023-07-19T14:06:39.034793Z"},"jupyter":{"outputs_hidden":false}}
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU device not found. Using CPU instead.')
    device_name = '/device:CPU:0'

# Assign the process to a GPU
with tf.device(device_name):
    # Train the model using the encoded quantum data
    history = model.fit(quantum_train_images, y_train, epochs=10, batch_size=32, validation_data=(quantum_val_images, y_val))

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T14:06:39.038097Z","iopub.execute_input":"2023-07-19T14:06:39.038647Z","iopub.status.idle":"2023-07-19T14:06:39.878739Z","shell.execute_reply.started":"2023-07-19T14:06:39.038618Z","shell.execute_reply":"2023-07-19T14:06:39.877646Z"},"jupyter":{"outputs_hidden":false}}
# Evaluate the accuracy of the trained model
loss, accuracy = model.evaluate(quantum_val_images, y_val)
print("Validation accuracy:", accuracy)

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"jupyter":{"outputs_hidden":false}}
# Define the classical neural network model using TensorFlow/Keras
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(2 ** num_qubits,)),
    layers.Dense(10, activation='softmax')
])

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T11:26:06.518680Z","iopub.execute_input":"2023-07-19T11:26:06.519749Z","iopub.status.idle":"2023-07-19T11:26:06.529020Z","shell.execute_reply.started":"2023-07-19T11:26:06.519708Z","shell.execute_reply":"2023-07-19T11:26:06.527700Z"},"jupyter":{"outputs_hidden":false}}

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import tensorflow as tf
from qiskit import Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.utils import QuantumInstance

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T11:26:09.787253Z","iopub.execute_input":"2023-07-19T11:26:09.787689Z","iopub.status.idle":"2023-07-19T11:27:27.880232Z","shell.execute_reply.started":"2023-07-19T11:26:09.787648Z","shell.execute_reply":"2023-07-19T11:27:27.879058Z"},"jupyter":{"outputs_hidden":false}}

# Load MNIST dataset
mnist = fetch_openml(name='mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T11:47:38.107184Z","iopub.execute_input":"2023-07-19T11:47:38.107651Z","iopub.status.idle":"2023-07-19T11:47:38.394791Z","shell.execute_reply.started":"2023-07-19T11:47:38.107592Z","shell.execute_reply":"2023-07-19T11:47:38.393639Z"},"jupyter":{"outputs_hidden":false}}
# Convert X dataset to float and normalize to range [0, 1]
X = X.astype(float) / 255.0

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T11:53:03.145787Z","iopub.execute_input":"2023-07-19T11:53:03.146987Z","iopub.status.idle":"2023-07-19T11:53:03.189029Z","shell.execute_reply.started":"2023-07-19T11:53:03.146944Z","shell.execute_reply":"2023-07-19T11:53:03.187604Z"},"jupyter":{"outputs_hidden":false}}
# Convert the target labels to one-hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T11:53:19.007494Z","iopub.execute_input":"2023-07-19T11:53:19.008636Z","iopub.status.idle":"2023-07-19T11:53:19.511873Z","shell.execute_reply.started":"2023-07-19T11:53:19.008571Z","shell.execute_reply":"2023-07-19T11:53:19.510742Z"},"jupyter":{"outputs_hidden":false}}
# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_onehot, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T11:48:14.619269Z","iopub.execute_input":"2023-07-19T11:48:14.620398Z","iopub.status.idle":"2023-07-19T11:48:14.628315Z","shell.execute_reply.started":"2023-07-19T11:48:14.620355Z","shell.execute_reply":"2023-07-19T11:48:14.626896Z"},"jupyter":{"outputs_hidden":false}}
# from qiskit import QuantumRegister, QuantumCircuit

# # Encode training and validation images using FRQI
# def frqi_encoding(image):
#     desired_length = 2 ** 4  # Length must be a power of 2
#     if len(image) < desired_length:
#         padded_image = np.pad(image, (0, desired_length - len(image)), 'constant')
#     else:
#         padded_image = image[:desired_length]
#     norm_value = np.linalg.norm(padded_image)
#     if norm_value == 0:
#         return padded_image
#     padded_image /= norm_value
#     return padded_image

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T11:48:28.040863Z","iopub.execute_input":"2023-07-19T11:48:28.041292Z","iopub.status.idle":"2023-07-19T11:48:33.734927Z","shell.execute_reply.started":"2023-07-19T11:48:28.041256Z","shell.execute_reply":"2023-07-19T11:48:33.733003Z"},"jupyter":{"outputs_hidden":false}}
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     print('GPU device not found. Using CPU instead.')
#     device_name = '/device:CPU:0'

# # Assign the process to a GPU
# with tf.device(device_name):
#     # Encode training and testing images using FRQI
#     frqi_train_images = [frqi_encoding(image) for image in X_train]
#     frqi_val_images = [frqi_encoding(image) for image in X_val]

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T11:56:04.452089Z","iopub.execute_input":"2023-07-19T11:56:04.452525Z","iopub.status.idle":"2023-07-19T11:56:04.923695Z","shell.execute_reply.started":"2023-07-19T11:56:04.452492Z","shell.execute_reply":"2023-07-19T11:56:04.922473Z"},"jupyter":{"outputs_hidden":false}}
# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_onehot, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T11:57:03.782524Z","iopub.execute_input":"2023-07-19T11:57:03.783577Z","iopub.status.idle":"2023-07-19T11:57:03.793375Z","shell.execute_reply.started":"2023-07-19T11:57:03.783535Z","shell.execute_reply":"2023-07-19T11:57:03.792013Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import tensorflow as tf
from qiskit import Aer, transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.utils import QuantumInstance

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:25:35.293300Z","iopub.execute_input":"2023-07-19T13:25:35.293794Z","iopub.status.idle":"2023-07-19T13:25:35.302714Z","shell.execute_reply.started":"2023-07-19T13:25:35.293751Z","shell.execute_reply":"2023-07-19T13:25:35.301479Z"},"jupyter":{"outputs_hidden":false}}
# Define quantum feature map and ansatz
num_qubits = 4
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
ansatz = TwoLocal(num_qubits, ['ry', 'rz'], 'cz')

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:25:37.211283Z","iopub.execute_input":"2023-07-19T13:25:37.211746Z","iopub.status.idle":"2023-07-19T13:25:37.218364Z","shell.execute_reply.started":"2023-07-19T13:25:37.211708Z","shell.execute_reply":"2023-07-19T13:25:37.216805Z"},"jupyter":{"outputs_hidden":false}}
# Define the quantum instance
quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:25:40.619172Z","iopub.execute_input":"2023-07-19T13:25:40.622261Z","iopub.status.idle":"2023-07-19T13:25:40.629313Z","shell.execute_reply.started":"2023-07-19T13:25:40.622214Z","shell.execute_reply":"2023-07-19T13:25:40.628014Z"},"jupyter":{"outputs_hidden":false}}
from qiskit.algorithms.optimizers import COBYLA

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:25:42.146400Z","iopub.execute_input":"2023-07-19T13:25:42.147940Z","iopub.status.idle":"2023-07-19T13:25:42.153635Z","shell.execute_reply.started":"2023-07-19T13:25:42.147890Z","shell.execute_reply":"2023-07-19T13:25:42.152019Z"},"jupyter":{"outputs_hidden":false}}
# Define the optimizer
optimizer = COBYLA(maxiter=100)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:25:44.193961Z","iopub.execute_input":"2023-07-19T13:25:44.194780Z","iopub.status.idle":"2023-07-19T13:25:44.282863Z","shell.execute_reply.started":"2023-07-19T13:25:44.194739Z","shell.execute_reply":"2023-07-19T13:25:44.281537Z"},"jupyter":{"outputs_hidden":false}}
# Build the Quantum SVM model
model = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    quantum_instance=quantum_instance
)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T13:25:49.075250Z","iopub.execute_input":"2023-07-19T13:25:49.076048Z","iopub.status.idle":"2023-07-19T13:25:49.136088Z","shell.execute_reply.started":"2023-07-19T13:25:49.075999Z","shell.execute_reply":"2023-07-19T13:25:49.133282Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.dataset_builders import split_dataset_to_data_and_labels
from qiskit.utils import QuantumInstance

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T12:12:54.411968Z","iopub.execute_input":"2023-07-19T12:12:54.412396Z","iopub.status.idle":"2023-07-19T12:12:54.467919Z","shell.execute_reply.started":"2023-07-19T12:12:54.412358Z","shell.execute_reply":"2023-07-19T12:12:54.466225Z"},"jupyter":{"outputs_hidden":false}}
# Convert data to data and labels format for Qiskit Aqua
train_data, _ = split_dataset_to_data_and_labels(X_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T12:32:26.907309Z","iopub.execute_input":"2023-07-19T12:32:26.907773Z","iopub.status.idle":"2023-07-19T12:32:27.473503Z","shell.execute_reply.started":"2023-07-19T12:32:26.907736Z","shell.execute_reply":"2023-07-19T12:32:27.471314Z"},"jupyter":{"outputs_hidden":false}}
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU device not found. Using CPU instead.')
    device_name = '/device:CPU:0'

# Assign the process to a GPU
with tf.device(device_name):
    
    
    # Train the model using the training data
    model.fit(X_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T08:44:50.196025Z","iopub.execute_input":"2023-07-19T08:44:50.196432Z","iopub.status.idle":"2023-07-19T08:44:50.695996Z","shell.execute_reply.started":"2023-07-19T08:44:50.196401Z","shell.execute_reply":"2023-07-19T08:44:50.694877Z"},"jupyter":{"outputs_hidden":false}}
# Reshape the statevectors to match the input shape of the neural network
frqi_train_statevectors = np.array(frqi_train_statevectors).reshape(-1, 4)
frqi_val_statevectors = np.array(frqi_val_statevectors).reshape(-1, 4)
frqi_test_statevectors = np.array(frqi_test_statevectors).reshape(-1, 4)

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T08:45:05.836176Z","iopub.execute_input":"2023-07-19T08:45:05.836547Z","iopub.status.idle":"2023-07-19T08:45:05.886740Z","shell.execute_reply.started":"2023-07-19T08:45:05.836515Z","shell.execute_reply":"2023-07-19T08:45:05.885731Z"},"jupyter":{"outputs_hidden":false}}
# Build a simple neural network for classification
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(4,)))  # 4 qubits for FRQI encoding
model.add(Dense(num_classes, activation='softmax'))  # Output layer with 10 classes

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T08:45:27.037408Z","iopub.execute_input":"2023-07-19T08:45:27.037815Z","iopub.status.idle":"2023-07-19T08:45:27.248445Z","shell.execute_reply.started":"2023-07-19T08:45:27.037783Z","shell.execute_reply":"2023-07-19T08:45:27.246546Z"},"jupyter":{"outputs_hidden":false}}
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU device not found. Using CPU instead.')
    device_name = '/device:CPU:0'

# Assign the process to a GPU
with tf.device(device_name):
    
    # Train the model on the training data
    model.fit(frqi_train_statevectors, y_train_onehot, epochs=10, batch_size=32, validation_data=(frqi_val_statevectors, y_val_onehot))

# %% [code] {"execution":{"iopub.status.busy":"2023-07-19T08:11:02.675431Z","iopub.status.idle":"2023-07-19T08:11:02.676195Z","shell.execute_reply.started":"2023-07-19T08:11:02.675925Z","shell.execute_reply":"2023-07-19T08:11:02.675950Z"},"jupyter":{"outputs_hidden":false}}
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU device not found. Using CPU instead.')
    device_name = '/device:CPU:0'

# Assign the process to a GPU
with tf.device(device_name):
    # Evaluate the model on the testing data
    accuracy = model.evaluate(np.array(frqi_test_statevectors), y_test_onehot)[1]
    print("Test Accuracy:", accuracy)

# %% [code] {"jupyter":{"outputs_hidden":false}}
