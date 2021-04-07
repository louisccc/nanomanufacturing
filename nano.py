import numpy as np
import cv2, statistics, pprint
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import sys
import csv
import matplotlib.pyplot as plt
import scipy.signal
import copy
import pyvisa as visa
import libusb1
import usb1
import time
import pyautogui

import queue

from sklearn.linear_model import LinearRegression


class Config:
    '''Configuration and Argument Parser for particle detection.'''
    def __init__(self, args):
        self.parser = ArgumentParser(description='parser for nanomanufacturing', formatter_class=RawTextHelpFormatter)
        self.parser.add_argument('--video_path', type=str, default=r"./10k20v.avi", help="Input path for video to image convertion")
        self.parser.add_argument('--image_path', type=str, default='./output', help="Output path for video to image convertion")
        self.parser.add_argument('--bar_image_path', type=str, default='./bar_output', help="Output path for video to image convertion")
        self.parser.add_argument('--detected_image_path', type=str, default='./detected_output', help="Output path for video to image convertion")
        self.parser.add_argument('--sampling_interval', type=int, default=30, help="Take one frame for every sampling interval.")
        self.parser.add_argument('--min_radius', type=int, default=1, help='Minimum radius for particles being detected.')
        self.parser.add_argument('--max_radius', type=int, default=30, help='Maximum radius for particles being detected.')
        self.parser.add_argument('--output_path_1', type=str, default='output1.txt', help="Output path for detected results in zone 1")
        self.parser.add_argument('--output_path_2', type=str, default='output2.txt', help="Output path for detected results in zone 2")
        self.parser.add_argument('--smooth_box_size', type=int, default=9, help="Size of the averaging box used for smoothing through 1D Convlution")
        self.parser.add_argument('--invalid_box_size', type=int, default=5, help="Size of the averaging box used for invalid data replacement")
        self.parser.add_argument('--param1', type=int, default=48, help="Parameter1 for HoughCircles()")
        self.parser.add_argument('--param2', type=int, default=25, help="Parameter2 for HoughCircles()")
        self.parser.add_argument('--output_time', type=int, default=0, help='Output x-axis in seconds instead of frame number')
        self.parser.add_argument('--polarity', type=int, default=1, help='Output detection result of positive or negative DEP')
        self.parser.add_argument('--polarity_limit', type=int, default=2, help='cut-off between moving and not moving for polarity detection')
        args_parsed = self.parser.parse_args(args)
        for arg_name in vars(args_parsed):
            self.__dict__[arg_name] = getattr(args_parsed, arg_name)

class ParticleDetector:
    def __init__(self, cfg):
        self.config = cfg
        self.video_path = cfg.video_path
        self.image_path = cfg.image_path
        self.bar_image_path = cfg.bar_image_path
        self.detected_image_path = cfg.detected_image_path
        self.sampling_interval = cfg.sampling_interval
        self.min_radius = cfg.min_radius
        self.max_radius = cfg.max_radius
        self.smooth_box_size = cfg.smooth_box_size
        self.invalid_box_size = cfg.invalid_box_size
        self.fps = 0
        self.path1 = cfg.output_path_1
        self.path2 = cfg.output_path_2

        self.xhat=0      # a posteri estimate of x
        self.P=0         # a posteri error estimate
        self.xhatminus=0 # a priori estimate of x
        self.Pminus=0    # a priori error estimate
        self.K=0         # gain or blending factor

        self.cap = cv2.VideoCapture(0)


    def convert_video_to_images(self, max_f_num = None):
        # sampling_interval = 30
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        max_frame_no = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if max_f_num == None else max_f_num
    
        # create a empty folder 
        root_folder_path = Path(self.image_path)
        root_folder_path.mkdir(parents=True, exist_ok=True)
        
        # capture images in avi file according to the framerate.
        for frame_no in range(0, max_frame_no, self.sampling_interval):
            if frame_no > 0  and frame_no < max_frame_no:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()
                outname = root_folder_path /  (str(frame_no)+'.jpg')
                cv2.imwrite(str(outname), frame)
        
        print("Video %s conversion successful." %str(self.video_path))

    def video_read(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        detected_image_folder_path = Path(self.detected_image_path)
        detected_image_folder_path.mkdir(parents=True, exist_ok=True)
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        

    def get_frame(self, frame_no):
        if frame_no < self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = self.cap.read()
            return frame

    def analyze_frame(self, frame, frame_no):
        INVALID_DATA = [-1, -1, -1, -1, -1, 0]
        bags_0 = []
        bags_1 = []
        detected_image_folder_path = Path(self.detected_image_path)

        img = frame
        kernel = np.ones((5,5),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,20, param1=self.config.param1,param2=self.config.param2,minRadius=self.min_radius, maxRadius=self.max_radius)
        try:
            detected_circles = np.uint16(np.around(detected_circles))         
            ''' filtering the detected circles out of area of interests '''
            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2]          
                
                if b < self.bar2[0][1] or b > self.bar2[1][1]: #filtering particles that are too high or too low.
                    continue
                if (a >= self.bar2[0][0] and a <= self.bar3[0][0]):
                    bags_0.append((a,b,r))
                if (a >= self.bar4[0][0] and a <= self.bar5[0][0]):
                    bags_1.append((a,b,r))
        except:
            dummy=0

        ''' draw bars and detected circles '''
        # cv2.line(frame, self.bar1[0], self.bar1[1], (0, 255, 0))
        cv2.line(frame, self.bar2[0], self.bar2[1], (0, 255, 0), 5)
        cv2.line(frame, self.bar3[0], self.bar3[1], (0, 255, 0), 5)
        # cv2.line(frame, self.bar4[0], self.bar4[1], (0, 255, 0))
        # cv2.line(frame, self.bar5[0], self.bar5[1], (0, 255, 0))    
        for a,b,r in bags_0+bags_1:
            cv2.circle(frame, (a, b), r, (0, 0, 255), 2) # Draw the circumference of the circle. 
            cv2.circle(frame, (a, b), 1, (255, 0, 0), 3) # Draw a small circle (of radius 1) to show the center. 
        out_path = "{}/{}.jpg".format(str(detected_image_folder_path), str(frame_no))
        cv2.imwrite(out_path, frame)
        cv2.imshow("Monitor", frame)

        ''' calculate features for each bead '''
        features_bag_0 = self.get_features_for_bag(bags_0, 2, 3)
        features_bag_1 = self.get_features_for_bag(bags_1, 4, 5)

        if features_bag_0 == INVALID_DATA:
            print("detecting beads having issues in zone 0: %s" % frame_no)
        if features_bag_1 == INVALID_DATA:
            print("detecting beads having issues in zone 1: %s" % frame_no)
        print("Particle detecting %s successful." % str(frame_no))
        return features_bag_0, features_bag_1

    def read_analyze_images(self):

        INVALID_DATA = [-1, -1, -1, -1, -1, 0]

        detected_image_folder_path = Path(self.detected_image_path)
        detected_image_folder_path.mkdir(parents=True, exist_ok=True)
        
        feature_group_0 = []
        feature_group_1 = []

        for image_path in Path(self.image_path).glob("**/*.jpg"):
            bags_0 = []
            bags_1 = []
            img = cv2.imread(str(image_path))
            kernel = np.ones((5,5),np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            detected_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,20, param1=self.config.param1,param2=self.config.param2,minRadius=self.min_radius, maxRadius=self.max_radius)
            try:
                detected_circles = np.uint16(np.around(detected_circles))         
                ''' filtering the detected circles out of area of interests '''
                for pt in detected_circles[0, :]: 
                    a, b, r = pt[0], pt[1], pt[2]          
                    
                    if b < self.bar1[0][1] or b > self.bar1[1][1]: #filtering particles that are too high or too low.
                        continue
                    if (a >= self.bar2[0][0] and a <= self.bar3[0][0]):
                        bags_0.append((a,b,r))
                    if (a >= self.bar4[0][0] and a <= self.bar5[0][0]):
                        bags_1.append((a,b,r))
            except:
                dummy=0

            ''' draw bars and detected circles '''
            cv2.line(gray, self.bar1[0], self.bar1[1], (0, 255, 0))
            cv2.line(gray, self.bar2[0], self.bar2[1], (0, 255, 0))
            cv2.line(gray, self.bar3[0], self.bar3[1], (0, 255, 0))
            cv2.line(gray, self.bar4[0], self.bar4[1], (0, 255, 0))
            cv2.line(gray, self.bar5[0], self.bar5[1], (0, 255, 0))    
            for a,b,r in bags_0+bags_1:
                cv2.circle(gray, (a, b), r, (0, 255, 0), 2) # Draw the circumference of the circle. 
                cv2.circle(gray, (a, b), 1, (0, 0, 255), 3) # Draw a small circle (of radius 1) to show the center. 
            cv2.imwrite(str(detected_image_folder_path / image_path.name), gray)

            ''' calculate features for each bead '''
            features_bag_0 = self.get_features_for_bag(bags_0, 2, 3)
            features_bag_1 = self.get_features_for_bag(bags_1, 4, 5)

            if features_bag_0 == INVALID_DATA:
                print("detecting beads having issues in zone 0: %s" % image_path)
            if features_bag_1 == INVALID_DATA:
                print("detecting beads having issues in zone 1: %s" % image_path)

            feature_group_0.append([int(image_path.name.split('.')[0]), features_bag_0])
            feature_group_1.append([int(image_path.name.split('.')[0]), features_bag_1])

            print("Particle detecting %s successful." % str(image_path))
        
        # sort result by frame number
        feature_group_0 = sorted(feature_group_0, key=lambda x: x[0])
        feature_group_1 = sorted(feature_group_1, key=lambda x: x[0])
        feature_group_0 = self.invalid_data_replacement(feature_group_0)
        feature_group_1 = self.invalid_data_replacement(feature_group_1)
        return feature_group_0, feature_group_1
    
    def invalid_data_replacement(self, features):
        INVALID_DATA = [-1, -1, -1, -1, -1, 0]
        num_feature = len(features[0][1])
        features_return = features
    
        for i in range(min([self.config.invalid_box_size, len(features_return)])):
            if features_return[i][1] == INVALID_DATA:
                features_return[i][1] = features_return[i - 1][1]

        for i in range(self.config.invalid_box_size, len(features_return)):
            if features_return[i][1] == INVALID_DATA:
                for avg_idx in range(num_feature):
                    avg_sum = 0
                    for avg_count in range(min([self.config.invalid_box_size, i])):
                        avg_sum = avg_sum + features_return[i - avg_count - 1][1][avg_idx]
                    avg_sum = avg_sum / self.config.invalid_box_size
                    features_return[i][1][avg_idx] = round(avg_sum, 3)
        
        return features_return 

    def draw_bars(self):
        ''' draw bars and stores the resulting images in bar_image_folder_path (for debug) '''
        bar_image_folder_path = Path(self.bar_image_path)
        bar_image_folder_path.mkdir(exist_ok=True)

        for image_path in Path(self.image_path).glob("**/*.jpg"):
            img = cv2.imread(str(image_path))
            cv2.line(img, self.bar1[0], self.bar1[1], (0, 255, 0))
            cv2.line(img, self.bar2[0], self.bar2[1], (0, 255, 0))
            cv2.line(img, self.bar3[0], self.bar3[1], (0, 255, 0))
            cv2.line(img, self.bar4[0], self.bar4[1], (0, 255, 0))
            cv2.line(img, self.bar5[0], self.bar5[1], (0, 255, 0))
            cv2.imwrite(str(bar_image_folder_path / image_path.name), img)

    def store_to_csv_files(self, path1, path2, feature_bag_0, feature_bag_1):
        with open(path1, 'w') as csvfile1:
            spamwriter = csv.writer(csvfile1)
            if self.config.output_time:
                spamwriter.writerow(["Sec","x_avr" ,"x_std"," avg_norm_x","avg_norm_abs_x","std_norm_abs_x","num_beads"])
            else:
                spamwriter.writerow(["Frames","x_avr" ,"x_std"," avg_norm_x","avg_norm_abs_x","std_norm_abs_x","num_beads"])
            for rows in feature_bag_0:
                temp_list = [rows[0]]
                temp_list[1:]= [i for i in rows[1]]
                if self.config.output_time:
                    temp_list[0] = round(temp_list[0]/self.fps, 2)
                # temp_list.insert(0, rows[0])
                spamwriter.writerow(temp_list)

        with open(path2, 'w') as csvfile2:
            spamwriter = csv.writer(csvfile2)
            if self.config.output_time:
                spamwriter.writerow(["Sec","x_avr" ,"x_std"," avg_norm_x","avg_norm_abs_x","std_norm_abs_x","num_beads"])
            else:
                spamwriter.writerow(["Frames","x_avr" ,"x_std"," avg_norm_x","avg_norm_abs_x","std_norm_abs_x","num_beads"])
            for rows in feature_bag_1:
                temp_list = [rows[0]]
                temp_list[1:]= [i for i in rows[1]]
                if self.config.output_time:
                    temp_list[0] = round(temp_list[0]/self.fps, 2)
                # temp_list.insert(0, rows[0])
                spamwriter.writerow(temp_list)

    def get_features_for_bag(self, bag, bar1, bar2):
        if bar1 == 2 and bar2 == 3:
            reference = self.bar2[0][0] + self.bar3[1][0]
        elif bar1 == 4 and bar2 == 5:
            reference = self.bar4[0][0] + self.bar5[1][0]
        else:
            reference = 0
        reference = reference/2

        try:
            x_avr = round(sum([p[0] for p in bag]) / len(bag), 3)
            x_std = round(statistics.stdev([float(p[0]) for p in bag]), 3)
            avg_norm_x = round(sum([(p[0] - reference) for p in bag]) / len(bag), 3)
            avg_norm_abs_x = round(sum([(abs(p[0] - reference)) for p in bag]) / len(bag), 3)
            std_norm_abs_x = round(statistics.stdev([float(abs(p[0] - reference)) for p in bag]), 3)

            num_beads = len(bag)

            return [x_avr, x_std, avg_norm_x, avg_norm_abs_x, std_norm_abs_x, num_beads]
        except: 
            return [-1, -1, -1, -1, -1, 0]

    def smooth_convolve(self, feature_bag):
        # feature_bag = sorted(feature_bag, key=lambda x: x[0])
        feature_return = copy.deepcopy(feature_bag)
        for smooth_idx in range(5):
            smooth_list = []
            for feature in feature_return:
                smooth_list.append(feature[1][smooth_idx])
            box = np.ones(self.smooth_box_size)/self.smooth_box_size
            smooth_output = np.convolve(smooth_list, box, mode='same')
            for i in range(int(self.smooth_box_size/2), int(len(feature_return)-(self.smooth_box_size/2))):
                feature_return[i][1][smooth_idx] = round(smooth_output[i],3)
        return feature_return

    def smooth_sg(self, feature_bag):
        feature_return = copy.deepcopy(feature_bag)
        for smooth_idx in range(5):
            smooth_list = []
            for feature in feature_return:
                smooth_list.append(feature[1][smooth_idx])
            smooth_output = scipy.signal.savgol_filter(smooth_list, 31, 3)
            for i in range(int(len(feature_return))):
                feature_return[i][1][smooth_idx] = round(smooth_output[i],3)
        return feature_return

    
    def check_bars(self):
        for image_path in Path(self.image_path).glob("**/*.jpg"):
            img = cv2.imread(str(image_path))
            height, width, channels = img.shape
            cv2.line(img, self.bar1[0], self.bar1[1], (0, 255, 0))
            cv2.line(img, self.bar2[0], self.bar2[1], (0, 255, 0))
            cv2.line(img, self.bar3[0], self.bar3[1], (0, 255, 0))
            cv2.line(img, self.bar4[0], self.bar4[1], (0, 255, 0))
            cv2.line(img, self.bar5[0], self.bar5[1], (0, 255, 0))
            for x_bars in range(0, height, 100):
                cv2.line(img, (0, x_bars),(width, x_bars), (255, 0, 0))
                cv2.putText(img, str(x_bars), (0, x_bars), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            for y_bars in range(0, width, 100):
                cv2.line(img, (y_bars, 0), (y_bars, height), (255, 0, 0))
                cv2.putText(img, str(y_bars), (y_bars, height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imshow('check_bars',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def check_polarity(self, feature_bag, feature_index, sampling_range=60):
        # This approach tries to find a regression line for each feature within a sampling range.
        # the slope of each regression are associated with the last feature in the range
        
        for i in range(sampling_range, len(feature_bag)):
            sampling_bag = []
            for bag_loop in range(sampling_range, 0, -1):
                sampling_bag.append(feature_bag[i - bag_loop][1][feature_index])
            # polynomial regression with order of 1. Return two values, Ax+B.
            result = np.polyfit(range(0, sampling_range * self.sampling_interval, self.sampling_interval), list(sampling_bag), 1)
            slope = result[0]
            feature_bag[i][1].append(slope)

            # # plot for testing
            # model = LinearRegression()
            # output_y = np.array(sampling_bag)
            # input_x = np.array(range(0, sampling_range * self.sampling_interval, self.sampling_interval)).reshape(-1,1)
            # model.fit(input_x, output_y)
            # all_features = []
            # for all_features_loop in range(len(feature_bag)):
            #     all_features.append(feature_bag[all_features_loop][1][feature_index])
            # # plt.scatter(range(0, len(feature_bag) * self.sampling_interval, self.sampling_interval), all_features, color='red')
            # plt.scatter(range(0, sampling_range * self.sampling_interval, self.sampling_interval), sampling_bag, color='red')
            # plt.plot(range(0, sampling_range * self.sampling_interval, self.sampling_interval), model.predict(input_x),color='blue')
            # plt.show()
            # import pdb; pdb.set_trace()

    def polarity(self, watching_window, feature_index, sampling_range=60):
        sampling_bag = []
        if len(watching_window) > sampling_range:
            for bag_loop in range(sampling_range, 0, -1):
                sampling_bag.append(watching_window[sampling_range - bag_loop][1][feature_index])
            result = np.polyfit(range(0, sampling_range * self.sampling_interval, self.sampling_interval), list(sampling_bag), 1)
        else:
            for bag_loop in range(len(watching_window), 0, -1):
                # import pdb; pdb.set_trace()
                sampling_bag.append(watching_window[len(watching_window) - bag_loop][1][feature_index])
            result = np.polyfit(range(0, len(watching_window) * self.sampling_interval, self.sampling_interval), list(sampling_bag), 1)

        return result[0]

def run_video_10k20v():
    ''' 
        only capture particles in the following zones: between bar2 and bar3, between bar4 and bar5.
    '''
    cfg = Config(sys.argv[1:])
    detector = ParticleDetector(cfg)
    detector.bar1 = [(265, 95), (265, 900)]
    detector.bar2 = [(490, 95), (490, 900)]
    detector.bar3 = [(710, 95), (710, 900)]
    detector.bar4 = [(936, 95), (936, 900)]
    detector.bar5 = [(1160, 95), (1160, 900)]
    detector.video_path = "./10k20v.avi"		
    detector.image_path = "./10k20v/extracted_frame"		
    detector.bar_image_path = "./10k20v/bar_frame"		
    detector.detected_image_path = "./10k20v/result_frame"

    detector.convert_video_to_images(max_f_num=6300)
    # detector.draw_bars()
    features_0, features_1 = detector.read_analyze_images()
    # smooth_features_0 = detector.smooth_convolve(features_0)
    detector.store_to_csv_files("./10k20v/output1.csv", "./10k20v/output2.csv", features_0, features_1)

def run_video_5mhz5v():
    ''' 
        only capture particles in the following zones: between bar2 and bar3, between bar4 and bar5.
    '''
    cfg = Config(sys.argv[1:])
    detector = ParticleDetector(cfg)
    detector.bar1 = [(1, 65), (1, 890)]
    detector.bar2 = [(250, 95), (250, 900)]
    detector.bar3 = [(500, 95), (500, 900)]
    detector.bar4 = [(965, 95), (965, 900)]
    detector.bar5 = [(1300, 95), (1300, 900)]
    detector.video_path = "./5mhz5v.avi"
    detector.image_path = "./5mhz5v_extracted_frame"
    detector.bar_image_path = "./5mhz5v_bar_frame"
    detector.detected_image_path = "./5mhz5v_result_frame"
    detector.path1 = "./5mhz5v_output1.txt"
    detector.path2 = "./5mhz5v_output2.txt"
    detector.convert_video_to_images()
    detector.check_bars()
    # detector.draw_bars()
    features_0, features_1 = detector.read_analyze_images()
    detector.store_to_csv_files("./5mhz5v_output1.csv", "./5mhz5v_output2.csv", features_0, features_1)

def run_video_2mhz1v():
    ''' 
        only capture particles in the following zones: between bar2 and bar3, between bar4 and bar5.
    '''
    cfg = Config(sys.argv[1:])
    detector = ParticleDetector(cfg)
    detector.bar1 = [(35, 5), (35, 800)]
    detector.bar2 = [(35, 5), (35, 800)]
    detector.bar3 = [(290, 5), (290, 800)]
    detector.bar4 = [(550, 5), (550, 800)]
    detector.bar5 = [(805, 5), (805, 800)]
    detector.video_path = "./2mhz1v.avi"		
    detector.image_path = "./2mhz1v(%s_%s)/extracted_frame" % (cfg.param1, cfg.param2)
    detector.bar_image_path = "./2mhz1v(%s_%s)/bar_frame" % (cfg.param1, cfg.param2)
    detector.detected_image_path = "./2mhz1v(%s_%s)/result_frame" % (cfg.param1, cfg.param2)

    detector.convert_video_to_images(max_f_num=27000)

    features_0, features_1 = detector.read_analyze_images()
    smooth_features_0 = detector.smooth_convolve(features_0)
    smooth_features_sg_0 = detector.smooth_sg(features_0)
    detector.check_polarity(smooth_features_0, 3)
    detector.store_to_csv_files("./2mhz1v(%s_%s)/output1.csv" % (cfg.param1, cfg.param2) , "./2mhz1v(%s_%s)/output2.csv" % (cfg.param1, cfg.param2), features_0, features_1)
    detector.store_to_csv_files("./2mhz1v(%s_%s)/output1_conv.csv" % (cfg.param1, cfg.param2) , "./2mhz1v(%s_%s)/output2_conv.csv" % (cfg.param1, cfg.param2), smooth_features_0, features_1)
    detector.store_to_csv_files("./2mhz1v(%s_%s)/output1_sg.csv" % (cfg.param1, cfg.param2) , "./2mhz1v(%s_%s)/output2_sg.csv" % (cfg.param1, cfg.param2), smooth_features_sg_0, features_1)

def run_video_20k_1v_layercage():
    ''' 
        only capture particles in the following zones: between bar2 and bar3, between bar4 and bar5.
    '''
    cfg = Config(sys.argv[1:])
    detector = ParticleDetector(cfg)
    detector.bar1 = [(40, 0), (40, 765)]
    detector.bar2 = [(40, 0), (40, 765)]
    detector.bar3 = [(285, 0), (285, 765)]
    detector.bar4 = [(550, 0), (550, 765)]
    detector.bar5 = [(800, 0), (800, 765)]
    output_folder = "./20k_1v(%s_%s)" % (cfg.param1, cfg.param2)
    detector.video_path = "./20k_1v_layercage.avi"	
    detector.image_path = "./20k_1v(%s_%s)/extracted_frame" % (cfg.param1, cfg.param2)	
    detector.bar_image_path = "./20k_1v(%s_%s)/bar_frame" % (cfg.param1, cfg.param2)	
    detector.detected_image_path = "./20k_1v(%s_%s)/result_frame" % (cfg.param1, cfg.param2)

    detector.convert_video_to_images()
    # detector.draw_bars()
    features_0, features_1 = detector.read_analyze_images()
    # smooth_features_0 = detector.smooth_convolve(features_0)
    detector.store_to_csv_files("./20k_1v(%s_%s)/output1.csv" % (cfg.param1, cfg.param2), "./20k_1v(%s_%s)/output2.csv" % (cfg.param1, cfg.param2), features_0, features_1)

def realtime_framework():
    # config setup
    cfg = Config(sys.argv[1:])
    detector = ParticleDetector(cfg)
    detector.bar1 = [(35, 5), (35, 800)]
    detector.bar2 = [(35, 5), (35, 800)]
    detector.bar3 = [(290, 5), (290, 800)]
    detector.bar4 = [(550, 5), (550, 800)]
    detector.bar5 = [(805, 5), (805, 800)]
    detector.video_path = "./2mhz1v.avi"
    detector.detected_image_path = "./2mhz1v(%s_%s)/result_frame" % (cfg.param1, cfg.param2)
    # input video

    total_frames = detector.video_read()
    result_zone0 = []
    result_zone1 = []
    frame_no = 0
    k = 30
    delta = 0.05
    watching_window = []
    watching_window_1 = []
    POSDEP = 0
    NEGDEP = 1 
    NOTMOVE= 2
    STATE = NOTMOVE
    while True:
        if frame_no > int(total_frames):
            break

        current_frame = detector.get_frame(frame_no)
        current_features_0, current_features_1 = detector.analyze_frame(current_frame, frame_no)
        watching_window.append([frame_no,current_features_0])
        watching_window_1.append([frame_no,current_features_1])

        # apply post-processing for data in watching_window.
        detector.invalid_data_replacement(watching_window)
        if(len(watching_window) > 10):
            slope = detector.polarity(watching_window, 4, 60)
            result_zone0.append(slope)
            if abs(float(slope)) <= delta:
                STATE = NOTMOVE
            elif slope > 0:
                STATE = POSDEP
            else:
                STATE = NEGDEP

            # TODO: According to the STATE, adjust volt/freq using function generator.
            if STATE == POSDEP:
                pass
                # increase freq by a number e.g. 100k
                # 
            elif STATE == NEGDEP:
                pass
                # decreae freq

        frame_no += detector.sampling_interval

    detector.store_to_csv_files("./2mhz1v(%s_%s)/output1.csv" % (cfg.param1, cfg.param2) , "./2mhz1v(%s_%s)/output2.csv" % (cfg.param1, cfg.param2), watching_window, watching_window_1)

def realtime_framework_screenshot():
    # config setup
    cfg = Config(sys.argv[1:])
    detector = ParticleDetector(cfg)
    detector.bar1 = [(35, 5), (35, 800)]
    detector.bar2 = [(200, 40), (200, 1040)]
    detector.bar3 = [(800, 40), (800, 1040)]
    detector.bar4 = [(550, 5), (550, 800)]
    detector.bar5 = [(805, 5), (805, 800)]
    output_folder = "./screen(%s_%s)" % (cfg.param1, cfg.param2)
    detector.detected_image_path = "./screen(%s_%s)/result_frame" % (cfg.param1, cfg.param2)
    
    # create a empty folder 
    root_folder_path = Path(output_folder)
    result_folder_path = Path(detector.detected_image_path)
    root_folder_path.mkdir(parents=True, exist_ok=True)
    result_folder_path.mkdir(parents=True, exist_ok=True)

    # screen recording with pyautogui. Save video for debugging purposes
    screenWidth, screenHeight = pyautogui.size() # get screen size
    resolution = (screenWidth, screenHeight)
    codec = cv2.VideoWriter_fourcc(*"XVID") # define codec
    filename = "./screen(%s_%s)/recording.avi" % (cfg.param1, cfg.param2)
    fps = 60.0 # FPS for video storage, not actual frame rate
    video_out = cv2.VideoWriter(filename, codec, fps, resolution) 
    cv2.namedWindow("Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Monitor", 480, 270)

    # FunctionGen setup
    # resources = visa.ResourceManager() #Establishes the resource (i.e equipment) manager from PyVISA
    # funcgen = resources.open_resource('USB0::0xF4EC::0xEE38::SDG2XCAC1L3169::INSTR') #Creates a name for the specific resource and opens the resource. The argument is the resource name specific to the function generator
    
    # AutoGUI setup
    # TODO: Automated software setup according to SOP

    result_zone0 = []
    result_zone1 = []
    frame_no = 0
    k = 30
    delta = 0.05
    watching_window = []
    watching_window_1 = []
    POSDEP = 0
    NEGDEP = 1 
    NOTMOVE= 2
    STATE = NOTMOVE
    while True:
        # Stop video recording by pressing q
        if cv2.waitKey(1) == ord('q'):
            break
        
        # get screen as video input
        current_screen = pyautogui.screenshot()
        current_frame = np.array(current_screen)
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB) # convert color from BGR to RGB
        video_out.write(current_frame) # store frame in the video for future reference
        # cv2.imshow("Monitor", current_frame)
        current_features_0, current_features_1 = detector.analyze_frame(current_frame, frame_no)
        watching_window.append([frame_no,current_features_0])
        watching_window_1.append([frame_no,current_features_1])

        # apply post-processing for data in watching_window.
        detector.invalid_data_replacement(watching_window)
        if(len(watching_window) > 40):
            slope = detector.polarity(watching_window, 4, 60)
            result_zone0.append(slope)
            if abs(float(slope)) <= delta:
                STATE = NOTMOVE
            elif slope > 0:
                STATE = POSDEP
            else:
                STATE = NEGDEP

            # TODO: According to the STATE, adjust volt/freq using function generator.
            if STATE == POSDEP:
                pass
                # increase freq by a number e.g. 100k
                # 
            elif STATE == NEGDEP:
                pass
                # decreae freq

        frame_no += 1

    detector.store_to_csv_files("./screen(%s_%s)/output1.csv" % (cfg.param1, cfg.param2) , "./screen(%s_%s)/output2.csv" % (cfg.param1, cfg.param2), watching_window, watching_window_1)
    # Stop video recording
    video_out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # run_video_10k20v()
    # run_video_5mhz5v()
    # run_video_2mhz1v()
    # run_video_20k_1v_layercage()

    # realtime_framework()
    realtime_framework_screenshot()