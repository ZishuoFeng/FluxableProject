import processing.serial.*;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.io.InputStreamReader;

Serial myPort;  // Create object from Serial class
String val;     // Data received from the serial port
boolean isRecording = false;
PrintWriter writer;
byte[] buffer = new byte[5];




void setup() 
{
  printArray(Serial.list());
  String portName = Serial.list()[1]; //change the 0 to a 1 or 2 etc. to match your port
  myPort = new Serial(this, portName, 115200);
}

void draw()
{
  if (isRecording && myPort.available() > 0) {
    //byte[] inByte = myPort.readBytes();
    //String data = String.format("%02X", inByte);
    //writer.println(data);
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
    }
  }
}

void keyPressed() {
  //Code for Calibration
  if(key == 'c' || key == 'C'){
    isRecording = !isRecording; // Toggle recording status
    if(isRecording){
      String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
      writer = createWriter("/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/" + "Calibration_" + timestamp + ".csv");
      
      writer.println("ID,Data1,Data2,Data3,Data4,Inductance_signal(Hex),Inductance_signal(Dec)");//Add the header of the table
      
      
    } else if(writer != null){
      writer.flush();
      writer.close();
      writer = null;
    }
  } 
  
  //Code for Extending
  else if(key == 'e' || key == 'E'){
    isRecording = !isRecording; // Toggle recording status
    if(isRecording){
      String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
      writer = createWriter("/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/" + "Extending_" + timestamp + ".csv");
      
      writer.println("ID,Data1,Data2,Data3,Data4,Inductance_signal(Hex),Inductance_signal(Dec)");//Add the header of the table 
      
      
    } else if(writer != null){
      writer.flush();
      writer.close();
      writer = null;
    }
  }
  
  
  //Code for bending
  else if(key == 'b' || key == 'B'){
    isRecording = !isRecording; // Toggle recording status
    if(isRecording){
      String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
      writer = createWriter("/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/" + "Bending_" + timestamp + ".csv");
      
      writer.println("ID,Data1,Data2,Data3,Data4,Inductance_signal(Hex),Inductance_signal(Dec)");//Add the header of the table 
      
      
    } else if(writer != null){
      writer.flush();
      writer.close();
      writer = null;
    }
  }
  
  //Code for Twisting
  else if(key == 't' || key == 'T'){
    isRecording = !isRecording; // Toggle recording status
    if(isRecording){
      String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
      writer = createWriter("/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/" + "Twisting_" + timestamp + ".csv");
      
      writer.println("ID,Data1,Data2,Data3,Data4,Inductance_signal(Hex),Inductance_signal(Dec)");//Add the header of the table 
      
      
    } else if(writer != null){
      writer.flush();
      writer.close();
      writer = null;
    }
  }
  
  //Code for compressing
  else if(key == 'o' || key == 'O'){
    isRecording = !isRecording; // Toggle recording status
    if(isRecording){
      String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
      writer = createWriter("/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/" + "Compressing_" + timestamp + ".csv");
      
      writer.println("ID,Data1,Data2,Data3,Data4,Inductance_signal(Hex),Inductance_signal(Dec)");//Add the header of the table 
      
      
    } else if(writer != null){
      writer.flush();
      writer.close();
      writer = null;
    }
  }
  
  //Code for pressing
  else if(key == 'p' || key == 'P'){
    isRecording = !isRecording; // Toggle recording status
    if(isRecording){
      String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
      writer = createWriter("/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/" + "Pressing_" + timestamp + ".csv");
      
      writer.println("ID,Data1,Data2,Data3,Data4,Inductance_signal(Hex),Inductance_signal(Dec)");//Add the header of the table 
      
      
    } else if(writer != null){
      writer.flush();
      writer.close();
      writer = null;
    }
  }
  
  //!!!!!not add the compound deformation yet!!!!!
  
  
  
  else if(key == 'r' || key == 'R'){
    try {
      Process process = Runtime.getRuntime().exec("/Users/asiu/Desktop/FluxableCode/t.py");//This python script needs to be changed. Please notice, the format in this script is "print()"
      BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
      String line;
      while ((line = reader.readLine()) != null) {
        println(line);
      }
      reader.close();
    } 
    catch (IOException e) {
      e.printStackTrace();
    }
  }
}
