import torch
import os
import pickle
import numpy as np
import pandas as pd

from  matplotlib import pyplot as plt
import matplotlib.image as mpimg

from milvus import Milvus, IndexType, MetricType, Status
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader



# def preprocess_images():
#     workers = 0 if os.name == 'nt' else 4
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     print('Running on device: {}'.format(device))

#     mtcnn = MTCNN(
#         image_size=160, margin=0, min_face_size=20,
#         thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=True,
#         device=device
#     )

#     resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


#     def collate_fn(x):
#         return x[0]

#     dataset = datasets.ImageFolder('/media/fico/Data/Celeba-low/img_align_celeba_organized/')
#     dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
#     loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)


#     encoded = []
#     identity = []
#     count = len(loader)

#     for x, y in loader:
#         try:
#             x_aligned, prob = mtcnn(x, return_prob=True)
#         except:
#             print(x)
#             plt.imshow(x)
#             plt.show()
#         if x_aligned is not None:
#             # x_aligned = torch.unsqueeze(x_aligned, dim=0)
#             x_aligned = x_aligned.to(device)
#             embeddings = resnet(x_aligned).detach().cpu()
#             embeddings = embeddings.numpy()
#             encoded.append(embeddings)
#             for x in range(embeddings.shape[0]):
#                 identity.append(dataset.idx_to_class[y])
#             if count %1000 == 0:
#                 print(count, x_aligned.shape, dataset.idx_to_class[y])
#             count -= 1
           
#     encoded = np.concatenate(encoded, 0)
#     encoded = np.squeeze(encoded)
#     print(encoded.shape)
#     identity = np.array(identity)
#     np.save("identity_save.npy", identity)
#     np.save("encoded_save.npy", encoded)
#     encoded = np.load("encoded_save.npy")
#     identity = np.load("identity_save.npy")
#     print(encoded.shape, identity.shape)


_HOST = '127.0.0.1'
_PORT = '19530' 
collection_name = 'celebrity_faces_'
id_to_identity = None
milvus = Milvus(_HOST, _PORT)

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=True,
        device=device
    )

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Vector parameters
_DIM = 512  # dimension of vector

_INDEX_FILE_SIZE = 32  # max file size of stored index

def create_collection():
    global id_to_identity
    print("Creating collection...")
    status, ok = milvus.has_collection(collection_name)
    if not ok:
        param = {
            'collection_name': collection_name,
            'dimension': _DIM,
            'index_file_size': _INDEX_FILE_SIZE,  # optional
            'metric_type': MetricType.L2  # optional
        }

        milvus.create_collection(param)
        print("Collection created.")
        return 1
    else:
        print("Collection present already.")
        try:
            with open ('id_to_class', 'rb') as fp:
                id_to_identity = pickle.load(fp)
            return 0
        except:
            return 0

def first_load():
    global id_to_identity
    print("Loading in encoded vectors...")
    encoded = np.load("encoded_save.npy")
    identity = np.load("identity_save.npy")

    encoded = np.array_split(encoded, 4, axis=0)
    identity = identity.astype(np.int)

    identity = np.array_split(identity, 4, axis=0)

    id_to_identity = []

    for x in range(len(encoded)):
        print(encoded[x].shape, encoded[x].dtype, identity[x])
        status, ids = milvus.insert(collection_name=collection_name, records=encoded[x])
        if not status.OK():
            print("Insert failed: {}".format(status))
        else:
            for z in range(len(ids)):
                id_to_identity.append((ids[z], identity[x][z]))

    # Flush collection  inserted data to disk.
    milvus.flush([collection_name])
    # # Get demo_collection row count
    # status, result = milvus.count_entities(collection_name)

    # # present collection statistics info
    # _, info = milvus.get_collection_stats(collection_name)
    with open('id_to_class', 'wb') as fp:
        pickle.dump(id_to_identity, fp)
    print("Vectors loaded in.")


def get_image_vectors(file_loc):
    img = Image.open(file_loc)
    bbx, prob = mtcnn.detect(img)
    embeddings = None
    if (bbx is not None):
        face_cropped = mtcnn.extract(img,bbx,None).to(device)
        embeddings = resnet(face_cropped).detach().cpu()
        embeddings = embeddings.numpy()
        draw = ImageDraw.Draw(img)
        for i, box in enumerate(bbx):
            draw.rectangle(box.tolist(), outline=(255,0,0))
            draw.text((box.tolist()[0] + 2,box.tolist()[1]), "Face-" + str(i), fill=(255,0,0))

    return embeddings, img

def index():
    print("Indexing...")
    # Obtain raw vectors by providing vector ids
    # create index of vectors, search more rapidly
    index_param = {
        'nlist': 4096
    }

    # Create ivflat index in demo_collection
    # You can search vectors without creating index. however, Creating index help to
    # search faster
    print("Creating index: {}".format(index_param))
    status = milvus.create_index(collection_name, IndexType.IVF_FLAT, index_param)

    # describe index, get information of index
    status, index = milvus.get_index_info(collection_name)
    print("Indexed.")

def search_image(file_loc):
    # # Use the top 10 vectors for similarity search
    query_vectors, insert_image = get_image_vectors(file_loc)


    # execute vector similarity search
    search_param = {
        "nprobe": 2056
    }

    print("Searching for image... ")

    param = {
        'collection_name': collection_name,
        'query_records': query_vectors,
        'top_k': 1,
        'params': search_param,
    }

    status, results = milvus.search(**param)
    if status.OK():
        temp = []
        plt.imshow(insert_image)
        for x in range(len(results)):
            for i, v in id_to_identity:
                if results[x][0].id == i:
                    temp.append(v)
        for i, x in enumerate(temp):
            fig = plt.figure()
            fig.suptitle('Face-' + str(i) + ", Celeb Folder: " + str(x))
            currentFolder = '/media/fico/Data/Celeba-low/img_align_celeba_organized/' + str(x)
            total = min(len(os.listdir(currentFolder)), 6)

            for i, file in enumerate(os.listdir(currentFolder)[0:total], 1):
                fullpath = currentFolder+ "/" + file
                # print(i, fullpath)
                img = mpimg.imread(fullpath)
                plt.subplot(2, 3, i)
                plt.imshow(img)
        plt.show(block = False)
        print(temp)

    # Delete demo_collection
def delete_collection():
    status = milvus.drop_collection(collection_name)

if __name__ == '__main__':
    # delete_collection()
    if create_collection():
        first_load()
    index()
    search_image("/media/fico/Data/Celeba-low/test2.jpg")
    plt.show()