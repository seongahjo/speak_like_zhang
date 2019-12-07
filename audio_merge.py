import argparse
import os


def concat_audio(prefix, sample_path, filters=' '):
	files = os.listdir(sample_path)
	files = list(filter(lambda f: f.startswith(prefix) and f.endswith('.wav'), files))
	files.sort()
	prefix = 'ffmpeg -f concat -i '
	file_name = 'temp.txt'
	with open(file_name, 'w') as f:
		for file in files:
			f.write('file {}\n'.format(os.path.join(sample_path, file)))

	command = '{} {} {} result.wav'.format(prefix, file_name, filters)
	print(command)
	os.system(command)
	os.remove(file_name)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--sample_path', default="logdir-tacotron2/generate")
	parser.add_argument('--speed', type=float, default='0.9')
	parser.add_argument('--prefix', required=True)
	config = parser.parse_args()
	all_filters = ' '
	if config.speed is not None:
		all_filters += '-af \"atempo={}\"'.format(str(config.speed))
	concat_audio(config.prefix, config.sample_path, all_filters)
