import controlP5.*;
import java.util.Map;
import processing.serial.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.io.InputStreamReader;
import toxi.geom.*;
import toxi.geom.mesh.*;

import toxi.processing.*;


ControlP5 cp5;
DropdownList deform_cat, MLM, n_estimators, max_depth, min_samples_split, learning_rate, C, kernel, gamma;
Button calibration, deformation, run, export, loadModel, nextItemButton;
Textarea resultArea;
Textlabel label_low, label_mid, label_high;
PVector modelPosition;
PShape myModel;

float modelRotation = 0;
Serial myPort;
float zoomFactor = 1;
float rotX, rotY;
PFont pfont;
ControlFont cfont;
CheckBox spring;
ArrayList<PShape> objModels = new ArrayList<PShape>();
ArrayList<String> springSection = new ArrayList<String>();
HashMap<String, PShape> map = new HashMap<String, PShape>();
HashMap<PShape, Integer> originalColors = new HashMap<PShape, Integer>();
HashMap<String, Button> deleteButtons = new HashMap<>();
int buttonCount = 0;




void setup() {
  size(1500, 800, P3D);

  
  cp5 = new ControlP5(this);
  
  pfont = createFont("Arial", 11);
  cfont = new ControlFont(pfont, 11);
  cp5.setFont(cfont);
  
  
  // Sidebar with controls
  loadModel = cp5.addButton("Load Model")
                 .setPosition(20, 10)
                 .setWidth(150)
                 .setHeight(30)
                 .addListener(new ControlListener() {
                       public void controlEvent(ControlEvent theEvent) { //<>//
                           selectFolder("Select a 3D model folder:", "folderSelected");
                       }
                   });
  loadModel.getCaptionLabel().toUpperCase(false);

  deform_cat = cp5.addDropdownList("Deformation Category")
                .setPosition(20, 120)
                .setWidth(150)
                .setHeight(150)
                .setOpen(false)
                .addItems(new String[]{"Compressing", "Extending", "Bending", "Twisting", "Twisting+Extending", "Twisting+Compressing", "Lateral pressing"})
                .addListener(new ControlListener() {
                    public void controlEvent(ControlEvent theEvent) {
                        DropdownList ddl = (DropdownList) theEvent.getController();
                        int selectedIndex = (int)ddl.getValue();
                        List<Map<String, Object>> items = ddl.getItems();
                        Map<String, Object> selectedItem = items.get(selectedIndex);
                        String itemName = (String) selectedItem.get("name");
                        onDeformCatItemSelected(itemName);
                        
                    }
                });
  deform_cat.getCaptionLabel().toUpperCase(false);
  deform_cat.getValueLabel().toUpperCase(false);



  MLM = cp5.addDropdownList("Machine Learning Model")
             .setPosition(20, 380)
             .setWidth(150)
             .setHeight(150)
             .setOpen(false)
             .addItems(new String[]{"Random Forest", "SVM", "XGB"});
  MLM.getCaptionLabel().toUpperCase(false);
  MLM.getValueLabel().toUpperCase(false);

  // Result area
  resultArea = cp5.addTextarea("Result")
                  .setPosition(210, height - 180)
                  .setSize(width - 220, (height - 630))
                  .setFont(createFont("Arial", 16))
                  .setLineHeight(18)
                  .setColor(color(128))
                  .setColorBackground(color(220))
                  .setColorForeground(color(255));

  modelPosition = new PVector(width * 0.75, height * 0.35);
  
  calibration = cp5.addButton("Calibration")
                   .setPosition(20, 320)
                   .setHeight(25)
                   .setWidth(150);
  calibration.getCaptionLabel().toUpperCase(false);
                   
                   
  deformation = cp5.addButton("Data Collection")
                   .setPosition(20, 350)
                   .setHeight(25)
                   .setWidth(150);
  deformation.getCaptionLabel().toUpperCase(false);
                   
                   
  run = cp5.addButton("Run the settings")
           .setPosition(20, 715)
           .setHeight(25)
           .setWidth(150);
  run.getCaptionLabel().toUpperCase(false);
           
  label_low = cp5.addTextlabel("label_low")
                 .setText("")
                 .setHeight(30)
                 .setPosition(20, 430)
                 .setColor(0);
  
  label_mid = cp5.addTextlabel("label_mid")
                 .setText("")
                 .setHeight(30)
                 .setPosition(20, 530)
                 .setColor(0)
                 .setHeight(20);
                          
  label_high = cp5.addTextlabel("label_high")
                  .setText("")
                  .setHeight(30)
                  .setPosition(20, 640)
                  .setColor(0)
                  .setHeight(20);
           
  n_estimators = cp5.addDropdownList("n_estimators")
                    .setPosition(20, 450)
                    .setWidth(150)
                    .setHeight(150)
                    .setOpen(false)
                    .hide()
                    .addItems(new String[]{"100","200","300","400","500"});
  n_estimators.getCaptionLabel().toUpperCase(false);
                    
  max_depth = cp5.addDropdownList("max_depth")
                  .setPosition(20, 550)
                  .setWidth(150)
                  .setHeight(150)
                  .setOpen(false)
                  .hide()
                  .addItems(new String[]{"2", "3", "5", "7", "10", "20"});
  max_depth.getCaptionLabel().toUpperCase(false);
                  
  min_samples_split = cp5.addDropdownList("min_samples_split")
                  .setPosition(20, 660)
                  .setWidth(150)
                  .setHeight(150)
                  .setOpen(false)
                  .hide()
                  .addItems(new String[]{"2", "5", "10"});
  min_samples_split.getCaptionLabel().toUpperCase(false);
                  
                  
  learning_rate = cp5.addDropdownList("learning rate")
                  .setPosition(20, 660)
                  .setWidth(150)
                  .setHeight(150)
                  .setOpen(false)
                  .hide()
                  .addItems(new String[]{"0.4", "0.1", "0.01", "0.001"});
  learning_rate.getCaptionLabel().toUpperCase(false);
                  
                  
  C = cp5.addDropdownList("C")
                  .setPosition(20, 450)
                  .setWidth(150)
                  .setHeight(150)
                  .setOpen(false)
                  .hide()
                  .addItems(new String[]{"0.1", "1", "10"});
                  
                  
  kernel = cp5.addDropdownList("kernel")
                  .setPosition(20, 550)
                  .setWidth(150)
                  .setHeight(150)
                  .setOpen(false)
                  .hide()
                  .addItems(new String[]{"linear", "rbf", "poly"});
  kernel.getCaptionLabel().toUpperCase(false);
                  
  gamma = cp5.addDropdownList("gamma")
                    .setPosition(20, 660)
                    .setWidth(150)
                    .setHeight(150)
                    .setOpen(false)
                    .hide()
                    .addItems(new String[]{"0.01", "0.1", "1"});
  gamma.getCaptionLabel().toUpperCase(false);
                    
                    
  export = cp5.addButton("Export the code")
         .setPosition(20, 760)
         .setHeight(25)
         .setWidth(150);
  export.getCaptionLabel().toUpperCase(false);
         
  nextItemButton = cp5.addButton("Next")
                      .setPosition(220, height - 50)
                      .hide()
                      .setSize(150, 25);
                      
  spring = cp5.addCheckBox("springPart")
              .setPosition(20, 50)
              .hide();
                   
                   
  printArray(Serial.list());
  String portName = Serial.list()[1]; //change the 0 to a 1 or 2 etc. to match your port
  myPort = new Serial(this, portName, 115200);
  
}

void draw() {
  background(240);
  

  // Simulated 3D model display (just a rotating rectangle for demo purposes)
  beginScissor(200, 200, width-200, height-200);
  lights();

  
  pushMatrix();
  camera(width/2.0, height/2.0, (height/2.0) / tan(PI*30.0 / 180.0), width/2.0, height/2.0, 0,0,1,0);
  translate(modelPosition.x, modelPosition.y);
  
  rotateX(rotX);
  rotateY(rotY);
  //noStroke();
  scale(zoomFactor);
  if(objModels != null){
    
    for(int i = 0;i < objModels.size();i++){
      String pathName = getKeyFromValue(map, objModels.get(i));
      if(pathName == null){
        shape(objModels.get(i));
      }else{
        if(spring.getState(pathName)){
        //  //resultArea.append(pathName);
          setPShapeColor(objModels.get(i), color(255,0,0,128));
          shape(objModels.get(i));
        }else{
          if(objModels.get(i).getChild(0).getFill(0) != originalColors.get(objModels.get(i))){
            setPShapeColor(objModels.get(i), originalColors.get(objModels.get(i)));
          }
          shape(objModels.get(i));
        }
      }
    }
  }
  

  popMatrix();
  endScissor();

  // Sidebar background divided into four sections
  fill(#3498DB);
  rect(0, 0, 200, 100);
  
  fill(#5DADE2);
  rect(0, 100, 200, 200);

  fill(#85C1E9);
  rect(0, 300, 200, 450);

  fill(#AED6F1);
  rect(0, 750, 200, 50);

  // Display area for 3D model simulation
  //fill(255);
  //rect(390, 0, width - 400, height - 190);
  
  ////Display the parameter console
  //fill(128);
  //rect(210, 0, 170, height - 190);
 
  
  
  if(shouldDrawRect){
    int i = 0;
    for(String s: selected_deform){ 
      fill(#D6EAF8);
      float w = 150;
      float h = 15;
      float x = 0;
      float y = 285 - i * h;
      rect(x,y,w,h);
      fill(0);
      textSize(10);
      textAlign(CENTER, CENTER);
      text(s, x+w/2, y+h/2);
      //Button temp_b = cp5.addButton("delete")
      //                   .setPosition(155, y)
      //                   .setSize(40,15)
      //                   .setColorBackground(color(255,0,0));
      i++;
    }
  }
  
  
  if (isRecording && myPort.available() > 0) {
    byte inByte = (byte)myPort.read();
    String data = String.format("%02X", inByte);
    println(data);
    for(int i = 0; i < buffer.length-1;i++){
      buffer[i] = buffer[i+1];
    }
    buffer[buffer.length - 1] = inByte;
    if(buffer[0] == 0x01){
      byte b1 = buffer[1];
      String d1 = String.format("%02X", b1);
      byte b2 = buffer[2];
      String d2 = String.format("%02X", b2);
      byte b3 = buffer[3];
      String d3 = String.format("%02X", b3);
      byte b4 = buffer[4];
      String d4 = String.format("%02X", b4);
      String inductance_signal_hex = d1 + d2 + d3 + d4;
      long inductance_signal_dec = Long.parseLong(inductance_signal_hex, 16);
      writer.println(buffer[0] + "," + d1 + "," + d2 + "," + d3 + "," + d4 + "," + inductance_signal_hex + "," + inductance_signal_dec);
      if(writer1 != null){
        writer1.println(buffer[0] + "," + d1 + "," + d2 + "," + d3 + "," + d4 + "," + inductance_signal_hex + "," + inductance_signal_dec);
      }
    }
  }
  
}

void beginScissor(int x, int y, int w, int h) {
    PGL pgl = beginPGL();
    pgl.enable(PGL.SCISSOR_TEST);
    pgl.scissor(x, y, w, h);
}

void endScissor() {
    PGL pgl = beginPGL();
    pgl.disable(PGL.SCISSOR_TEST);
}


void folderSelected(File selection) {
  if (selection == null) {
    resultArea.setText("No folder was selected.");
    return;
  } else {
    resultArea.setText("User selected " + selection.getAbsolutePath());
    
    // Filter and load all .obj files in the folder
    File[] objFiles = selection.listFiles();

    for (File objFile : objFiles) {
      if(objFile.getName().contains(".obj")){
        PShape shape = loadShape(objFile.getAbsolutePath());
        originalColors.put(shape, shape.getChild(0).getFill(0));///////////////////////////////////////////////////////////////////////////////
        if(objFile.getName().contains("spring")){
            springSection.add(objFile.getName());
            map.put(objFile.getName(), shape);
        }
      objModels.add(shape);
      }
    }
    for(int i = 0; i < springSection.size();i++){
      spring.addItem(springSection.get(i), i+1);
    }
    spring.show();
    resultArea.append(objModels.size() + " .obj files loaded.");
  }
}

public String getKeyFromValue(HashMap<String, PShape> map, PShape value) {
    for (String key : map.keySet()) {
        if (value.equals(map.get(key))) {
            return key;
        }
    }
    return null;
}

void setPShapeColor(PShape s, int c) {
  for (int i = 0; i < s.getVertexCount(); i++) {
    s.setFill(c);
  }
  
  for (int i = 0; i < s.getChildCount(); i++) {
    setPShapeColor(s.getChild(i), c);
  }
}



void onDeformCatItemSelected(String item) {
    
    //String selectedItem = theEvent.getStringValue(); 

    
    Button deleteButton = cp5.addButton("delete_" + item)
                             .setPosition(155, 285 - buttonCount * 15)
                             .setSize(45,15)
                             .setLabel("delete")
                             .setColorBackground(color(255,0,0))
                             .addListener(new ControlListener() {
                                  public void controlEvent(ControlEvent theEvent) {
                                      removeItem(item);
                                      //resultArea.append(item + "\n");
                                  }
                              });
    deleteButtons.put(item, deleteButton);
    //resultArea.append("\n" + deleteButtons.toString());
    buttonCount++;
}

void removeItem(String item) {
    
    selected_deform.remove(item);

    
    Button buttonToRemove = deleteButtons.remove(item);
    //resultArea.append("\n" + buttonToRemove);
    buttonToRemove.hide();
    cp5.remove(buttonToRemove);
    buttonCount--;
}
