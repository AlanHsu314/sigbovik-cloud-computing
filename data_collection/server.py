from flask import Flask, send_file, jsonify

app = Flask(__name__)

img_template = 'data/images/%s.png'
gt_template = 'data/GTmaps/%s_GT.png'

@app.route('/img/<int:idx>')
def render_image(idx):
    idx = str(idx).rjust(4, '0')
    return send_file(img_template % idx, mimetype='image/png')

@app.route('/gt_map/<int:idx>')
def render_gt_map(idx):
    idx = str(idx).rjust(4, '0')
    return send_file(gt_template % idx, mimetype='image/png')

@app.route('/annotate/<int:idx>/<label>', methods=['POST'])
def annotate(idx, label):
    with open('annotations.txt', 'a') as f:
        f.write(f"{idx},{label}\n")
    return jsonify({'ok': 'ok'})

app.run(port=5000)