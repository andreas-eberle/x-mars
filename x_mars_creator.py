import argparse
import csv
import glob
import os
import random
import shutil
from collections import OrderedDict


def get_unique_market_labels(market_files):
	labels = [os.path.basename(name).split('_')[0] for name in market_files]
	labels_int = [int(label) for label in labels]
	labels_unique = set(labels_int)
	return labels_unique


def get_mars_label_for_filename(filename):
	label_string = os.path.basename(filename)[0:4]
	return int(label_string) if label_string != '00-1' else -1


def get_mars_cam_for_filename(filename):
	cam_string = os.path.basename(filename)[5]
	return int(cam_string)


def get_mars_track_for_filename(filename):
	track_string = os.path.basename(filename)[7:11]
	return int(track_string)


def get_unique_mars_labels(mars_files):
	labels = [get_mars_label_for_filename(name) for name in mars_files]
	return set(labels)


def write_x_mars_tracks_test(test_by_label_cam_track, output_directory):
	file_counter = 1
	tracks_counter = 1

	with open(os.path.join(output_directory, 'x-mars-tracks-test.csv'), 'w', newline='') as tracks_file, \
			open(os.path.join(output_directory, 'x-mars-test-name.txt'), 'w', newline='') as names, \
			open(os.path.join(output_directory, 'x-mars-query-idx.csv'), 'w', newline='') as queries_file:
		tracks_writer = csv.writer(tracks_file, delimiter=',')
		queries_writer = csv.writer(queries_file, delimiter=',')

		for label, by_cams in OrderedDict(sorted(test_by_label_cam_track.items())).items():
			for cam, by_tracks in OrderedDict(sorted(by_cams.items())).items():
				for track, files in OrderedDict(sorted(by_tracks.items())).items():
					start = file_counter
					file_counter += len(files)
					end = file_counter - 1

					tracks_writer.writerow([start, end, label, cam])

					for file in sorted(files):
						names.write(os.path.basename(file) + '\n')

				if label > 0 and len(by_cams) > 1:  # no queries for 0 and -1
					query_track_idx = tracks_counter + random.randint(0, len(by_tracks) - 1)
					queries_writer.writerow([query_track_idx])

				tracks_counter += len(by_tracks)


def write_x_mars_tracks_train(train_by_label_cam_track, output_directory):
	file_counter = 1
	tracks_counter = 1

	with open(os.path.join(output_directory, 'x-mars-tracks-train.csv'), 'w', newline='') as tracks_file, \
			open(os.path.join(output_directory, 'x-mars-train-name.txt'), 'w', newline='') as names:
		tracks_writer = csv.writer(tracks_file, delimiter=',')

		for label, by_cams in OrderedDict(sorted(train_by_label_cam_track.items())).items():
			for cam, by_tracks in OrderedDict(sorted(by_cams.items())).items():
				for track, files in OrderedDict(sorted(by_tracks.items())).items():
					start = file_counter
					file_counter += len(files)
					end = file_counter - 1

					tracks_writer.writerow([start, end, label, cam])

					for file in sorted(files):
						names.write(os.path.basename(file) + '\n')

				tracks_counter += len(by_tracks)


def create_x_mars(market_directory, mars_directory, output_directory):
	#
	# Read all market files
	market_test_all_files = glob.glob(os.path.join(market_directory, 'bounding_box_test/*.jpg'))
	market_test_labels_unique = get_unique_market_labels(market_test_all_files)

	market_train_all_files = glob.glob(os.path.join(market_directory, 'bounding_box_train/*.jpg'))
	market_train_labels_unique = get_unique_market_labels(market_train_all_files)

	print('Market Test Labels')
	print(market_test_labels_unique)
	print('Market Train Labels')
	print(market_train_labels_unique)

	#
	# Read all mars files

	mars_test_files = glob.glob(os.path.join(mars_directory, 'bbox_test/*/*.jpg'))
	mars_test_labels = get_unique_mars_labels(mars_test_files)

	mars_train_files = glob.glob(os.path.join(mars_directory, 'bbox_train/*/*.jpg'))
	mars_train_labels = get_unique_mars_labels(mars_train_files)

	print('MARS Test and Train Labels')
	mars_all_labels_unique = mars_test_labels.union(mars_train_labels)
	print(mars_all_labels_unique)

	#
	# Calculate X-MARS test/train split

	x_mars_test_labels = market_test_labels_unique.intersection(mars_all_labels_unique)
	print('Mars + Market Test intersection Labels')
	print(x_mars_test_labels)

	x_mars_train_labels = market_train_labels_unique.intersection(mars_all_labels_unique)
	print('Mars + Market Train intersection Labels')
	print(x_mars_train_labels)

	print()

	#
	# Create X-MARS file lists

	mars_all_files = []
	mars_all_files.extend(mars_test_files)
	mars_all_files.extend(mars_train_files)

	x_mars_test_files = []
	x_mars_train_files = []

	for file in mars_all_files:
		label = get_mars_label_for_filename(file)
		if label in x_mars_test_labels:
			x_mars_test_files.append(file)
		else:
			x_mars_train_files.append(file)

	#
	# Create Dictionary with X-MARS files by label, cam and track

	x_mars_test_by_label_cam_track = {}

	for file in x_mars_test_files:
		label = get_mars_label_for_filename(file)
		cam = get_mars_cam_for_filename(file)
		track = get_mars_track_for_filename(file)

		x_mars_test_by_label_cam_track.setdefault(label, {}).setdefault(cam, {}).setdefault(track, []).append(file)

	x_mars_train_by_label_cam_track = {}

	for file in x_mars_train_files:
		label = get_mars_label_for_filename(file)
		cam = get_mars_cam_for_filename(file)
		track = get_mars_track_for_filename(file)

		x_mars_train_by_label_cam_track.setdefault(label, {}).setdefault(cam, {}).setdefault(track, []).append(file)

	# Write track files

	write_x_mars_tracks_test(x_mars_test_by_label_cam_track, output_directory)
	write_x_mars_tracks_train(x_mars_train_by_label_cam_track, output_directory)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--market', help='Directory with market dataset')
	parser.add_argument('--mars', help='Directory with mars dataset')
	parser.add_argument('--output', help='Directory where to write x-mars files')
	args = parser.parse_args()

	print('Running with command line arguments:')
	print(args)
	print('\n\n')

	output_directory = args.output

	if os.path.exists(output_directory):
		shutil.rmtree(output_directory)
	os.makedirs(output_directory)

	create_x_mars(args.market, args.mars, output_directory)


if __name__ == '__main__':
	main()
