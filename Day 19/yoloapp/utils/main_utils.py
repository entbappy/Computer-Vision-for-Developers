import base64 

def encodeImagetoBase64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read())
    


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open("data/" + fileName, 'wb') as f:
        f.write(imgdata)
        f.close()