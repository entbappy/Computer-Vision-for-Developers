from yoloapp.utils.main_utils import encodeImagetoBase64, decodeImage


encode_img = encodeImagetoBase64("data/hello.jpg")

decodeImage(encode_img, "hello_decode.jpg")