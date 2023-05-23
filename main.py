# face verification with the VGGFace2 model
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = Image.open(filename)
    midCoord = (int(pixels.size[0] / 2), int(pixels.size[1] / 2))
    image = pixels.crop((0, midCoord[1] - 800, midCoord[0] + 500, midCoord[1] + 1000))
    face = image.resize(required_size)
    face = face.rotate(270)
    face_array = asarray(face)
    return face_array



# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.3):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    with open('result.txt', 'w') as resultFile:
        if score <= thresh:
            print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
        else:
            print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))


# define filenames
templateFilenames = []
exampleFilenames = []

with open('template.txt', 'r') as file1:
    lines = file1.readlines()
    for line in lines:
        templateFilenames.append(line)
with open('example.txt', 'r') as file2:
    lines = file2.readlines()
    for line in lines:
        exampleFilenames.append(line)

# get embeddings file filenames
templateEmbeddings = get_embeddings(templateFilenames)
exampleEmbeddings = get_embeddings(exampleFilenames)


for face in exampleEmbeddings:
    for template in templateEmbeddings:
        is_match(template, face)
