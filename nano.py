import numpy as np
import cv2, statistics, pprint
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
import os
import sys
import csv

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
        self.parser.add_argument('--max_radius', type=int, default=50, help='Maximum radius for particles being detected.')
        self.parser.add_argument('--output_path_1', type=str, default='output1.txt', help="Output path for detected results in zone 1")
        self.parser.add_argument('--output_path_2', type=str, default='output2.txt', help="Output path for detected results in zone 2")
        self.parser.add_argument('--smooth_box_size', type=int, default=5, help="Size of the averaging box used for smoothing through 1D Convlution")
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

        self.path1 = cfg.output_path_1
        self.path2 = cfg.output_path_2

    def convert_video_to_images(self, max_f_num = None):
        # sampling_interval = 30
        cap = cv2.VideoCapture(self.video_path)
        max_frame_no = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if max_f_num == None else max_f_num
    
        # create a empty folder 
        root_folder_path = Path(self.image_path)
        root_folder_path.mkdir(exist_ok=True)
        
        # capture images in avi file according to the framerate.
        for frame_no in range(0, max_frame_no, self.sampling_interval):
            if frame_no > 0  and frame_no < max_frame_no:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()
                outname = root_folder_path /  (str(frame_no)+'.jpg')
                cv2.imwrite(str(outname), frame)
        
        print("Video %s conversion successful." %str(self.video_path))

    def read_analyze_images(self):

        detected_image_folder_path = Path(self.detected_image_path)
        detected_image_folder_path.mkdir(exist_ok=True)
        
        feature_group_0 = []
        feature_group_1 = []
        feature_group_normalized_0 = []
        feature_group_normalized_1 = []

        for image_path in Path(self.image_path).glob("**/*.jpg"):
            bags_0 = []
            bags_1 = []
            img = cv2.imread(str(image_path))
            kernel = np.ones((5,5),np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_blurred = cv2.blur(gray, (3,3))

            detected_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,20, param1=50,param2=30,minRadius=self.min_radius, maxRadius=self.max_radius)
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

            ''' draw bars and detected circles '''
            cv2.line(img, self.bar1[0], self.bar1[1], (0, 255, 0))
            cv2.line(img, self.bar2[0], self.bar2[1], (0, 255, 0))
            cv2.line(img, self.bar3[0], self.bar3[1], (0, 255, 0))
            cv2.line(img, self.bar4[0], self.bar4[1], (0, 255, 0))
            cv2.line(img, self.bar5[0], self.bar5[1], (0, 255, 0))    
            for a,b,r in bags_0+bags_1:
                cv2.circle(img, (a, b), r, (0, 255, 0), 2) # Draw the circumference of the circle. 
                cv2.circle(img, (a, b), 1, (0, 0, 255), 3) # Draw a small circle (of radius 1) to show the center. 
            cv2.imwrite(str(detected_image_folder_path / image_path.name), img)

            ''' calculate features for each bead '''
            features_bag_0 = self.get_features_for_bag(bags_0, 2, 3)
            features_bag_1 = self.get_features_for_bag(bags_1, 4, 5)

            if features_bag_0 == [-1, -1, -1, -1, -1, 0]:
                print("detecting beads having issues in zone 0: %s" % image_path)
            if features_bag_1 == [-1, -1, -1, -1, -1, 0]:
                print("detecting beads having issues in zone 1: %s" % image_path)

            feature_group_0.append([int(image_path.name.split('.')[0]), features_bag_0])
            feature_group_1.append([int(image_path.name.split('.')[0]), features_bag_1])

            print("Particle detecting %s successful." % str(image_path))
        
        # sort result by frame number
        feature_group_0 = sorted(feature_group_0, key=lambda x: x[0])

        # # remove invalid data
        # for i in range(len(feature_group_0)):
        #     if feature_group_0[i][1] == [-1, -1, -1, -1, -1, 0]:
        #         if i != 0:
        #             feature_group_0[i][1] = feature_group_0[i - 1][1] 
        #         else:
        #             feature_group_0[i][1] = feature_group_0[i + 1][1] 
        #     # import pdb; pdb.set_trace()

        avg_box = 5
        for i in range(len(feature_group_0)):
            if feature_group_0[i][1] == [-1, -1, -1, -1, -1, 0]:
                if i >= avg_box:
                    for avg_idx in range(5):
                        avg_sum = 0
                        for avg_count in range(avg_box):
                            avg_sum = avg_sum + feature_group_0[i - avg_count - 1][1][avg_idx]
                        avg_sum = avg_sum / avg_box
                        feature_group_0[i][1][avg_idx] = round(avg_sum, 3)
                else:
                    feature_group_0[i][1] = feature_group_0[i + 1][1]

        return feature_group_0, feature_group_1
        
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

    def store_to_txt_files(self, path1, path2, feature_bag_0, feature_bag_1):
        feature_bag_0 = sorted(feature_bag_0, key=lambda x: x[0])
        feature_bag_1 = sorted(feature_bag_1, key=lambda x: x[0])
        
        with open(path1, 'w') as file1, open(path2, 'w') as file2:
            pprint.pprint(feature_bag_0, file1)
            pprint.pprint(feature_bag_1, file2)

    def store_to_csv_files(self, path1, path2, feature_bag_0, feature_bag_1):
        with open(path1, 'w') as csvfile1:
            spamwriter = csv.writer(csvfile1)
            spamwriter.writerow(["Frames","x_avr" ,"x_std"," avg_norm_x","avg_norm_abs_x","std_norm_abs_x","num_beads"])
            for rows in feature_bag_0:
                temp_list = [rows[0]]
                temp_list[1:]= [i for i in rows[1]]
                # temp_list.insert(0, rows[0])
                spamwriter.writerow(temp_list)

        with open(path2, 'w') as csvfile2:
            spamwriter = csv.writer(csvfile1)
            spamwriter.writerow(["Frames","x_avr" ,"x_std"," avg_norm_x","avg_norm_abs_x","std_norm_abs_x","num_beads"])
            for rows in feature_bag_1:
                temp_list = [rows[0]]
                temp_list[1:]= [i for i in rows[1]]
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
        for smooth_idx in range(5):
            smooth_list = []
            for feature in feature_bag:
                smooth_list.append(feature[1][smooth_idx])
            box = np.ones(self.smooth_box_size)/self.smooth_box_size
            smooth_output = np.convolve(smooth_list, box, mode='same')
            for i in range(int(self.smooth_box_size/2), int(len(feature_bag)-(self.smooth_box_size/2))):
                feature_bag[i][1][smooth_idx] = round(smooth_output[i],3)
        return feature_bag

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
    detector.image_path = "./10k20v_extracted_frame"		
    detector.bar_image_path = "./10k20v_bar_frame"		
    detector.detected_image_path = "./10k20v_result_frame"

    detector.convert_video_to_images(max_f_num=6300)
    # detector.draw_bars()
    features_0, features_1 = detector.read_analyze_images()
    # smooth_features_0 = detector.smooth_convolve(features_0)
    # detector.store_to_txt_files("./10k20v_output1.txt", "./10k20v_output2.txt", features_0, features_1)
    detector.store_to_csv_files("./10k20v_output1.csv", "./10k20v_output2.csv", features_0, features_1)

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
    detector.draw_bars()
    features_0, features_1 = detector.read_analyze_images()
    detector.store_to_txt_files("./5mhz5v_output1.txt", "./5mhz5v_output2.txt", features_0, features_1)

def run_video_2mhz1v():
    ''' 
        only capture particles in the following zones: between bar2 and bar3, between bar4 and bar5.
    '''
    cfg = Config(sys.argv[1:])
    detector = ParticleDetector(cfg)
    detector.bar1 = [(50, 95), (50, 750)]
    detector.bar2 = [(285, 5), (285, 750)]
    detector.bar3 = [(570, 5), (570, 750)]
    detector.bar4 = [(770, 5), (770, 750)]
    detector.bar5 = [(970, 5), (970, 750)]
    detector.video_path = "./2mhz1v.avi"		
    detector.image_path = "./2mhz1v_extracted_frame"		
    detector.bar_image_path = "./2mhz1v_bar_frame"		
    detector.detected_image_path = "./2mhz1v_result_frame"

    detector.convert_video_to_images(max_f_num=6300)
    # detector.draw_bars()
    features_0, features_1 = detector.read_analyze_images()
    # smooth_features_0 = detector.smooth_convolve(features_0)
    detector.store_to_txt_files("./2mhz1v_output1.txt", "./2mhz1v_output2.txt", features_0, features_1)

if __name__ == "__main__":
    # run_video_10k20v()
    # run_video_5mhz5v()
    run_video_2mhz1v()