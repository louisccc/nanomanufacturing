import numpy as np
import cv2, statistics
from pathlib import Path
import pprint

def convert_video_to_images(avi_path):
    frame_rate = 30
    sampling_interval = 30
    cap = cv2.VideoCapture(avi_path)
    max_frame_no = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   
    # create a empty folder 
    root_folder_path = Path('./output')
    root_folder_path.mkdir(exist_ok=True)
    
    # capture images in avi file according to the framerate.
    for frame_no in range(0, max_frame_no, sampling_interval):
        if frame_no > 0  and frame_no < max_frame_no:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            outname = root_folder_path /  (str(frame_no)+'.jpg')
            cv2.imwrite(str(outname), frame)


bar1 = [(265, 95), (265, 900)]
bar2 = [(490, 95), (490, 900)]
bar3 = [(710, 95), (710, 900)]
bar4 = [(936, 95), (936, 900)]
bar5 = [(1160, 95), (1160, 900)]

def read_analyze_images(output_path):
    # should be a 4-tuple item list with each 4-tuple (x1, y1, x2, y2)
    # only capture particles in the following zones: between bar2 and bar3, between bar4 and bar5.
    feature_group_0 = []
    feature_group_1 = []

    for image_path in Path(output_path).glob("**/*.jpg"):
        bags_0 = []
        bags_1 = []
        img = cv2.imread(str(image_path))
        kernel = np.ones((5,5),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,20, param1=50,param2=30,minRadius=1, maxRadius=50)
        detected_circles = np.uint16(np.around(detected_circles)) 
        
        cv2.line(img, bar1[0], bar1[1], (0, 255, 0))
        cv2.line(img, bar2[0], bar2[1], (0, 255, 0))
        cv2.line(img, bar3[0], bar3[1], (0, 255, 0))
        cv2.line(img, bar4[0], bar4[1], (0, 255, 0))
        cv2.line(img, bar5[0], bar5[1], (0, 255, 0))

        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2]          
            
            if b < bar1[0][1] or b > bar1[1][1]: #filtering particles that are too high or too low.
                continue
            if (a >= bar2[0][0] and a <= bar3[0][0]):
                bags_0.append((a,b,r))
            if (a >= bar4[0][0] and a <= bar5[0][0]):
                bags_1.append((a,b,r))
            
        for a,b,r in bags_0+bags_1:
            cv2.circle(img, (a, b), r, (0, 255, 0), 2) # Draw the circumference of the circle. 
            cv2.circle(img, (a, b), 1, (0, 0, 255), 3) # Draw a small circle (of radius 1) to show the center. 
        
        
        
        # perform particle detections?
        # calculate the average x and y per zone.
        # store it back to a list
        
        try:
            x_avr_0 = sum([p[0] for p in bags_0]) / len(bags_0)
            x_std_0 = statistics.stdev([float(p[0]) for p in bags_0])
            y_avr_0 = sum([p[1] for p in bags_0]) / len(bags_0)
            y_std_0 = statistics.stdev([float(p[1]) for p in bags_0])

            x_avr_1 = sum([p[0] for p in bags_1]) / len(bags_1)
            x_std_1 = statistics.stdev([float(p[0]) for p in bags_1])
            y_avr_1 = sum([p[1] for p in bags_1]) / len(bags_1)
            y_std_1 = statistics.stdev([float(p[1]) for p in bags_1])

            feature_group_0.append((int(image_path.name.split('.')[0]), (x_avr_0, x_std_0, y_avr_0, y_std_0)))
            feature_group_1.append((int(image_path.name.split('.')[0]), (x_avr_1, x_std_1, y_avr_1, y_std_1)))
        except:
            print(image_path)
            cv2.imshow("Detected Circle", img) 
            cv2.waitKey(0) 

    feature_group_0 = sorted(feature_group_0, key=lambda x: x[0])
    feature_group_1 = sorted(feature_group_1, key=lambda x: x[0])
    
    with open('output1.txt', 'w') as file1, open('output2.txt', 'w') as file2:
        pprint.pprint(feature_group_0, file1)
        pprint.pprint(feature_group_1, file2)
    import pdb; pdb.set_trace()
if __name__ == "__main__":
    avi_path = r"./10k20v.avi" 
    # convert_video_to_images(avi_path)
    read_analyze_images("./output")