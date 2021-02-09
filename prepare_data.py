import os
import sys
import zipfile
import pathlib
import shutil

def download():
    # img_url = "https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing"
    img_url = "https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    # ident_url = "https://drive.google.com/file/d/1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS/view?usp=sharing"
    output = 'celeba.zip'
    ident_url = "https://drive.google.com/uc?id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS"

def unzip():
    file = "./img_align_celeba.zip"
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall()
def reorganize():
    file = open("./identity_CelebA.txt")
    run = True
    while(run):
        line = file.readline()
        if not line:
            run = False
        else:
            line = line.strip()
            split = line.split(" ")
            img = split[0]
            ident = split[1]
            pathlib.Path("./celeb_reorganized/" + str(ident)).mkdir(parents = True, exist_ok=True)
            from_loc = pathlib.Path('./img_align_celeba/' + str(img))
            to_loc = pathlib.Path('./celeb_reorganized/' + str(ident) + "/" + str(img))
            shutil.copy(from_loc, to_loc) 
    shutil.rmtree("./img_align_celeba")
if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("Incorret arguments: Arg1=ident_file, Arg2: Picture directory")
    # else:
    print("Unzipping Data...")
    unzip()
    print("Reorganizing Data...")
    reorganize()
