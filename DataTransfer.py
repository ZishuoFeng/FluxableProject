import serial
import csv
import time

s = serial.Serial('/dev/cu.usbmodem11401', 9600)

filename = 'data.csv'
# Table Header
column_names = ['ID', 'Timestamp', 'Data']



# Open CSV file in read-only mode and read existing lines
with open(filename, mode='r', newline='') as f:
    f.seek(0,0)
    reader = csv.reader(f)
    num_existing_rows = sum(1 for _ in reader)
    print(num_existing_rows)


with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    flag = 0
    # if this csv file is blank, add the table header
    if num_existing_rows == 0:
        writer.writerow(column_names)
    else:
        flag = 1

    # Move the pointer to the end of this file in order to add data
        # 2 means the end of the file; 0 means the head; 1 means the current
        # 0 means the offset according to the location according to the second parameter
    file.seek(0, 2)
    temp = 0

    while True:
        i = num_existing_rows + temp + 1 - flag
        data = s.readline().decode('utf-8').rstrip('\r\n')
        if data:
            timestamp = time.time()
            writer.writerow([i, timestamp, data])
            temp += 1

s.close()