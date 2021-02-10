What is Milvus:

Milvus is a vector similarity search engine, a tool that lets you quickly find the closest matching vector in a pool of billions of vectors. It does this by bringing together many different indexing algorithms under one roof, allowing users to test and choose which algorithms work best for their current needs. In order to speed these searches up even faster, Milvus can leverage GPUs for speedup in large index searches.


What is the value of Milvus:

Milvus makes large vectors search queries fast, and most importantly simple and customizable. With ML and AI research reaching a point where it is possible to generate meaningful vector embeddings, the challenge of being able to search through billions of these embeddings quickly comes to play. Milvus solves this, allowing companies to spend less time on figuring out how to store and search through this data, and instead focus on how to improve the data they are generating.

The Milvus User Experience:

The Milvus python implementation is one of the easiest database implementations that I have used. Milvus manages to keep it simple while still offering a high level of customization. Although my project is on the simpler end, I have yet to run into any bugs or error messages that I could not understand. One of the key features I enjoy is the quick ability to test out different configurations and indexing methods, as changing these requires the change of only a few parameters.



Scenario:

One big use case for Milvus is for facial recognition. Facial recognition requires searching through millions of faces, with each face capable of having hundreds of separate embeddings. In order for this to be effective, it needs to be fast and scalable, both of which Milvus are.

My implementation scales this down a bit and tries to find the celebrities that are in a photo/scene. It works by using a pytorch implementation of MTCNN and Facenet. MTCNN locates all the faces in the images and passes it to Facenet to create an embedding of each face. With this vector we can find the closest match within a Milvus database of encoded celebrity faces. The result shows a few of the faces of the celebrity and a corresponding numeric code because the dataset of faces does not include names, but rather id tags.


How to Run:
1. Setup and run Milvus with default host/port: 

https://www.milvus.io/docs/install_milvus.md/

2. Clone this github repo: 

git clone https://github.com/fhaltmayer/milvus.git 

3. Download each of these files and place them into the downloaded github repo:

https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/edit 

https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS/view?usp=sharing

4. Optional but recommended: Download the encoded celebrity files into the repo (will save a lot of time due to slow processing of images)

https://drive.google.com/file/d/1kWRApLKWveCHsdVH2TCNF2GPKRYw2ZdO/view?usp=sharing 

https://drive.google.com/file/d/1KlgWG8pClNgX1LU2hvbgRBZQSFg_50ey/view?usp=sharing 

5. Install requirements: pip install -r requirments.txt

6. Run: python celeb_finder.py filepath_of_picture 
