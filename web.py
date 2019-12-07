import argparse

from flask import Flask, request, render_template, jsonify, \
	send_from_directory, make_response, send_file
import os
import json
from synthesizer import Synthesizer
from utils import str2bool

AUDIO_DIR = "audio"
AUDIO_PATH = os.path.join(AUDIO_DIR)

app = Flask(__name__)


@app.route('/generate', methods=['GET'])
def generate():
	print(request.args.decode('utf-8'))
	text = json.loads(request.args.decode('utf-8'))["text"]
	print(text)
	audio_path = synthesizer.synthesize(texts=text.split('\n'), base_path='logdir-tacotron2/generate', speaker_ids=[0],
										attention_trim=True, base_alignment_path=None,
										isKorean=True, config=config)[0]
	return send_file(audio_path, mimetype="audio/wav",
					 as_attachment=True,
					 attachment_filename='file.wav')


@app.route('/', methods=['GET'])
def rendering():
	addr = request.url_root
	return render_template('view.html', addr=addr)


parser = argparse.ArgumentParser()
parser.add_argument('--sample_path', default="logdir-tacotron2/generate")
parser.add_argument('--num_speakers', default=1, type=int)
parser.add_argument('--speaker_id', default=0, type=int)
parser.add_argument('--checkpoint_step', default=None, type=int)
parser.add_argument('--base_alignment_path', default=None)
parser.add_argument('--file', type=str, default='test.wav')
parser.add_argument('--gpu', type=int, default=0)
config = parser.parse_args()
synthesizer = Synthesizer()
synthesizer.load('model', 1, None, inference_prenet_dropout=False, config=config)

app.run(host='0.0.0.0', port=80)
