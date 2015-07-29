import os
import StringIO
from PIL import Image
from werkzeug import secure_filename
from scipy.misc import imread, imresize
from caption_generator import generate_captions
from flask import Flask, request, render_template


UPLOAD_FOLDER = '/tmp/'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = 'super_secret_key'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS


@app.route("/", methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filename)
                captions, scores = generate_captions(filename)
                return render_template('captions.html',
                                       filename=filename,
                                       scores=scores,
                                       captions=captions,
                                       imagesrc=embed_image_html(filename))
            return app.send_static_file('error.html')
        return app.send_static_file('index.html')
    except Exception as e:
        print e
        return app.send_static_file('error.html')


def embed_image_html(filename):
    im = imread(filename)
    im = imresize(im, (254, 254))
    image_pil = Image.frombuffer('RGB', (254, 254), im.tostring(), 'raw', 'RGB', 0, 1)
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)