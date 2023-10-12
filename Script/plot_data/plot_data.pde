ArrayList <PVector> pointList = new ArrayList <PVector> ();
IntList posList = new IntList();
Table table;

String file1 = "/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/Compressing_20230820_231924.csv";
String file2 = "/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/Calibration_20230814_001800.csv";


int posRecorded = 0;
PVector max = new PVector();
PVector min = new PVector();

PFont mono;


void setup() {
  size (2000, 800);
  readSignalCSVFile();
  processPoints();
  
  mono = createFont("/Users/asiu/Desktop/FluxableCode/Processing/signal_label/data/andalemo.ttf", 30);
  textFont(mono);

  smooth();
}
 
void draw() {
  background(0);
  stroke(210);
  fill(210, 90);
  int pidx = 0;
  for (;pidx <= pointList.size()-1; pidx++) {
    if(pidx == 0){
      ellipse(pointList.get(pidx).x, pointList.get(pidx).y, 1, 1);
    }
    else{
      ellipse(pointList.get(pidx).x, pointList.get(pidx).y, 1, 1);
      line(pointList.get(pidx).x, pointList.get(pidx).y, pointList.get(pidx-1).x, pointList.get(pidx-1).y);
    }
  }
  
  if(posList.size() > 0){
    int idx = 0;
    for(;idx <= posList.size() - 1; idx++){
      stroke(72,164,222);
      fill(72,164,222,50);
      
      if(idx%2 == 0){
        // segment start
        line(posList.get(idx), 0, posList.get(idx), height);
      }
      else{
        // segment end
        line(posList.get(idx), 0, posList.get(idx), height);
        rect(posList.get(idx-1), 0, posList.get(idx) - posList.get(idx-1), height);
      }
      
      int actualPos = int(map(posList.get(idx), 0, width, min.x, max.x)) + 1;
      fill(210,90);
      text(str(actualPos), posList.get(idx), height/2);
    }
  }
  
  stroke(210);
  line(mouseX, 0, mouseX, height);
}
 
void readSignalCSVFile() {
  table = loadTable(file1, "header");
  println(table.getRowCount() + " total rows in table");
   String[] headers = table.getColumnTitles();
  for (String header : headers) {
    println(header);
  }
  float count = 1;
  for (TableRow row : table.rows()) {
    float xx = count;
    long yy = row.getLong("Inductance_signal(Dec)");
    float zz = 0.0f;
    pointList.add(new PVector(xx, yy, zz));
    count++;
  }
}
 
void mouseClicked() {
  posRecorded = mouseX;
  posList.append(posRecorded);
}

void processPoints() {  
  min.x = pointList.get(0).x;
  min.y = pointList.get(0).y;
  max.x = pointList.get(0).x;
  max.y = pointList.get(0).y;
  
  for (PVector p : pointList) {
    if (p.x > max.x) { max.x = p.x; }
    if (p.y > max.y) { max.y = p.y; }
    
    if (p.x < min.x) { min.x = p.x; }
    if (p.y < min.y) { min.y = p.y; }
  }
  println("Min_x: ", min.x);
  println("Max_x: ", max.x);
  println("Min_y: ", min.y);
  println("Max_y: ", max.y);
  // adapt values to screen
  for (PVector p : pointList) {
    //map the value of p.x from the range (min.x, max.x) to range (0, width)
    p.x = map(p.x, min.x, max.x, 0, width);
    p.y = map(p.y, min.y, max.y, 0, height);
    //reverse the y axis upside down
    p.y = height - p.y;
  }
}
