This is a project that helps you figure out which celebrities are in a photo/scene. It works by using a pytorch implementation of MTCNN and Facenet. MTCNN locates all the faces in the images and passes it to Facenet to create a latent encoding of each face. With this tensor we can find the closest match within a milvus database of encoded celebrity faces. The result shows a few of the faces of the celebrity and a corresponding code because the dataset of faces does not include names, but rather id tags. 

How to Run:
1. Setup and run Milvus: 

https://www.milvus.io/docs/install_milvus.md/

2. Clone this github repo: git clone 

https://github.com/fhaltmayer/milvus.git 

3. Download each of these files and place them into the downloaded github repo:

https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/edit 

https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS/view?usp=sharing

4. Optional but recommended: Download the encoded celebrity files into the repo (will save a lot of time due to slow processing of images)

https://drive.google.com/file/d/1kWRApLKWveCHsdVH2TCNF2GPKRYw2ZdO/view?usp=sharing 

https://drive.google.com/file/d/1KlgWG8pClNgX1LU2hvbgRBZQSFg_50ey/view

5. Run: celeb_finder.py filepath_of_picture 
