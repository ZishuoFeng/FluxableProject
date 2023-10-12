boolean isRecording = false;
PrintWriter writer, writer1;
byte[] buffer = new byte[5];


void keyPressed() {
  
  //Calibration
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
  
  //Compressing
  if(key == 'o' || key == 'O'){
    isRecording = !isRecording; // Toggle recording status
    if(isRecording){
      String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
      ArrayList<String> selectedSpringSection = new ArrayList<String>();
      for(int i = 0;i < springSection.size();i++){
        String pathName = springSection.get(i);
        if(spring.getState(pathName)){
          selectedSpringSection.add(pathName.replace(".obj", ""));
        }
      }
      
      String springName = "";
      if(selectedSpringSection.size() == 1){
        springName = selectedSpringSection.get(0) + "_";
        writer = createWriter("/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/" + "Compressing_" + springName + timestamp + ".csv");
        writer.println("ID,Data1,Data2,Data3,Data4,Inductance_signal(Hex),Inductance_signal(Dec)");//Add the header of the table
      }else{
        writer = createWriter("/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/" + "Compressing_" + selectedSpringSection.get(0) + "_" + timestamp + ".csv");
        writer.println("ID,Data1,Data2,Data3,Data4,Inductance_signal(Hex),Inductance_signal(Dec)");//Add the header of the table
        writer1 = writer = createWriter("/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/" + "Compressing_" + selectedSpringSection.get(1) + "_" + timestamp + ".csv");
        writer1.println("ID,Data1,Data2,Data3,Data4,Inductance_signal(Hex),Inductance_signal(Dec)");//Add the header of the table
      }
      
      
      
    } else if(writer != null){
      writer.flush();
      writer.close();
      writer = null;
    }else if(writer1 != null){
      writer.flush();
      writer.close();
      writer = null;
    }
  }
  
  //Extending
  if(key == 'e' || key == 'E'){
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
  
  //Twisting
  if(key == 't' || key == 'T'){
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
  
  //Twisting+Extending
  if(key == 'r' || key == 'R'){
    isRecording = !isRecording; // Toggle recording status
    if(isRecording){
      String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
      writer = createWriter("/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/" + "Twisting&Extending_" + timestamp + ".csv");
      writer.println("ID,Data1,Data2,Data3,Data4,Inductance_signal(Hex),Inductance_signal(Dec)");//Add the header of the table
    } else if(writer != null){
      writer.flush();
      writer.close();
      writer = null;
    }
  }
  
  //Twisting+Compressing
  if(key == 'f' || key == 'F'){
    isRecording = !isRecording; // Toggle recording status
    if(isRecording){
      String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
      writer = createWriter("/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/" + "Twisting&Compressing_" + timestamp + ".csv");
      writer.println("ID,Data1,Data2,Data3,Data4,Inductance_signal(Hex),Inductance_signal(Dec)");//Add the header of the table
    } else if(writer != null){
      writer.flush();
      writer.close();
      writer = null;
    }
  }
  
  //Lateral pressing
  if(key == 'l' || key == 'L'){
    isRecording = !isRecording; // Toggle recording status
    if(isRecording){
      String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
      writer = createWriter("/Users/asiu/Desktop/FluxableCode/DataLogs/LiangDeformations/" + "LateralPressing_" + timestamp + ".csv");
      writer.println("ID,Data1,Data2,Data3,Data4,Inductance_signal(Hex),Inductance_signal(Dec)");//Add the header of the table
    } else if(writer != null){
      writer.flush();
      writer.close();
      writer = null;
    }
  }
  
  //Bending
  if(key == 'b' || key == 'B'){
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
}
