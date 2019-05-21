import csv
import cv2



# (train/test, filename, class, xmin	ymin	xmax	ymax)
to_process = []

with open('input/labels/ftc_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            to_process.append(('train', row[0], row[3], (int(row[4]), int(row[5]),int(row[6]),int(row[7]))))

            if line_count % 5 is 0:
                to_process.append(('train', row[0], "none", (10, 10,50,50)))
                to_process.append(('train', row[0], "none", (80, 10,140,50)))
                to_process.append(('train', row[0], "none", (100, 10,150, 60)))

            line_count += 1
    print(f'Processed {line_count} lines.')

with open('input/labels/ftc_test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            to_process.append(('test', row[0], row[3], (int(row[4]), int(row[5]),int(row[6]),int(row[7]))))

            if line_count % 5 is 0:
                to_process.append(('test', row[0], "none", (10, 10,50,50)))
                to_process.append(('test', row[0], "none", (80, 10,140,50)))
                to_process.append(('test', row[0], "none", (100, 10,150, 60)))
    print(f'Processed {line_count} lines.')

save_index = 0;
for entry in to_process:
    sec = entry[0]
    class_name = entry[2]
    image_name = entry[1] 
    box = entry[3]
    img = cv2.imread('input/images/' + image_name)
  
    crop_img = img[box[1]:box[3], box[0]:box[2]]
    print(crop_img.shape)
    save_index += 1;
    cv2.imwrite("output/" + sec + "/" + class_name +"/"+str(save_index) + ".jpg", crop_img)
    
