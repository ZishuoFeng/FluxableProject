Set<String> selected_deform = new HashSet<String>();
boolean shouldDrawRect, shouldDrawModelTree;
List<String> svm_param = new ArrayList<>();
List<String> rf_param = new ArrayList<>();
List<String> xgb_param = new ArrayList<>();
List<String> code = new ArrayList<>();
String selectedModel, c, ker, gam, n, max_d, min_s, learning;
ArrayList<String> remainingList;

void controlEvent(ControlEvent theEvent) {
  if(theEvent.isFrom(loadModel)){
    for(int i = 0; i < springSection.size();i++){
      spring.addItem(springSection.get(i), i+1);
    }
    spring.show();
  }
  
  
  
  if (theEvent.isFrom(deform_cat)) {
    // Handle deform_cat checkbox logic
    int index = (int)theEvent.getValue();
    Map<String, Object> selectedItem = deform_cat.getItem(index);
    String selectedItemName = (String)selectedItem.get("name");
    if(selected_deform.contains(selectedItemName)){
      selected_deform.remove(selectedItemName);
    }else{
      selected_deform.add(selectedItemName);
    }
    shouldDrawRect = true;
    remainingList = new ArrayList<>(selected_deform);
  }
  
  //Handle MLM dropdown logic
  if (theEvent.isFrom(MLM)) {
    int index = (int)theEvent.getValue();
    Map<String, Object> selectedItem = MLM.getItem(index);
    String selectedItemName = (String)selectedItem.get("name");
    selectedModel = selectedItemName;
    if(resultArea.getText().contains("Machine Learning Model")){
      resultArea.clear();
    }
    if("Random Forest" == selectedItemName){
      label_low.setText("n_estimator");
      label_mid.setText("max_depth");
      label_high.setText("min_samples_split");
      learning_rate.hide();
      C.hide();
      kernel.hide();
      gamma.hide();
      n_estimators.show();
      max_depth.show();
      min_samples_split.show();
    }
    if("XGB" == selectedItemName){
      label_low.setText("n_estimator");
      label_mid.setText("max_depth");
      label_high.setText("leanring_rate");
      min_samples_split.hide();
      C.hide();
      kernel.hide();
      gamma.hide();
      n_estimators.show();
      max_depth.show();
      learning_rate.show();
    }
    if("SVM" == selectedItemName){
      label_low.setText("C");
      label_mid.setText("kernel");
      label_high.setText("gamma");
      n_estimators.hide();
      max_depth.hide();
      min_samples_split.hide();
      C.show();
      kernel.show();
      gamma.show();
    }
    resultArea.setText("Selected Machine Learning Model: " + selectedItemName + "\n" + "Please choose the parameters for this model. " + "\n");
  }
  
  if(theEvent.isFrom(calibration)){
    resultArea.setText("Press c to continue your calibration. ");
  }
  
  if(theEvent.isFrom(deformation)){
    showRandomItem();
    nextItemButton.show();
  }
  
  if(theEvent.isFrom(n_estimators)){
    int index = (int)theEvent.getValue();
    Map<String, Object> selectedItem = n_estimators.getItem(index);
    String selectedItemName = (String)selectedItem.get("name");
    n = selectedItemName;
  }
  
  if(theEvent.isFrom(C)){
    int index = (int)theEvent.getValue();
    Map<String, Object> selectedItem = C.getItem(index);
    String selectedItemName = (String)selectedItem.get("name");
    c = selectedItemName;
  }
  
  if(theEvent.isFrom(max_depth)){
    int index = (int)theEvent.getValue();
    Map<String, Object> selectedItem = max_depth.getItem(index);
    String selectedItemName = (String)selectedItem.get("name");
    max_d = selectedItemName;
  }
  
  if(theEvent.isFrom(min_samples_split)){
    int index = (int)theEvent.getValue();
    Map<String, Object> selectedItem = min_samples_split.getItem(index);
    String selectedItemName = (String)selectedItem.get("name");
    min_s = selectedItemName;
  }
  
  if(theEvent.isFrom(learning_rate)){
    int index = (int)theEvent.getValue();
    Map<String, Object> selectedItem = learning_rate.getItem(index);
    String selectedItemName = (String)selectedItem.get("name");
    learning = selectedItemName;
  }
  
  if(theEvent.isFrom(kernel)){
    int index = (int)theEvent.getValue();
    Map<String, Object> selectedItem = kernel.getItem(index);
    String selectedItemName = (String)selectedItem.get("name");
    ker = selectedItemName;
  }
  
  if(theEvent.isFrom(gamma)){
    int index = (int)theEvent.getValue();
    Map<String, Object> selectedItem = gamma.getItem(index);
    String selectedItemName = (String)selectedItem.get("name");
    gam = selectedItemName;
  }
  
  if(theEvent.isFrom(nextItemButton)){
    showRandomItem();
  }
  
  if(theEvent.isFrom(run)){
    try {
      
      String num = Integer.toString(selected_deform.size());
      String exec_path = "/Users/asiu/Desktop/FluxableCode/Test.py";
      if("Random Forest" == selectedModel){
        exec_path = exec_path + " " + "Random_Forest" + " " + n + " " + max_d + " " + min_s + " " + num;
      }
      if("XGB" == selectedModel){
        exec_path = exec_path + " " + "XGB" + " " + n + " " + max_d + " " + learning + " " + num;
      }
      if("SVM" == selectedModel){
        exec_path = exec_path + " " + "SVM" + " " + c + " " + ker + " " + gam + " " + num;
      }
      Process process = Runtime.getRuntime().exec(exec_path);//Add more parameter to transfer them into python document. Starting from sys.argv[1]. 
      BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));

      BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
      String line;
      while ((line = reader.readLine()) != null) {
        //println(line);
        resultArea.append(line + "\n");
      }
      
      String errorLine;
      while ((errorLine = errorReader.readLine()) != null) {
          resultArea.append("[ERROR] " + errorLine + "\n");
      }
      errorReader.close();
      
      
      reader.close();
    } 
    catch (IOException e) {
      e.printStackTrace();
    }
  }
  
 
  
  
  if(theEvent.isFrom(export)){
    if(selectedModel == "SVM"){
      try {
          code.clear();
          Process process = Runtime.getRuntime().exec("/Users/asiu/Desktop/FluxableCode/export_svm.py");//Add more parameter to transfer them into python document. Starting from sys.argv[1]. 
          BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
    
          BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
          String line;
          while ((line = reader.readLine()) != null) {
            code.add(line);
          }
          String[] codes = code.toArray(new String[code.size()]);
          saveStrings("exported_svm.txt", codes);
          resultArea.setText("Finish exporting! You can find the code in a .txt file at the current path. ");
          
          String errorLine;
          while ((errorLine = errorReader.readLine()) != null) {
              resultArea.append("[ERROR] " + errorLine + "\n");
          }
          errorReader.close();
          
          
          reader.close();
        } 
        catch (IOException e) {
          e.printStackTrace();
        }
    }
    
    if(selectedModel == "Random Forest"){
      try {
          code.clear();
          Process process = Runtime.getRuntime().exec("/Users/asiu/Desktop/FluxableCode/export_rf.py");//Add more parameter to transfer them into python document. Starting from sys.argv[1]. 
          BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
    
          BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
          String line;
          while ((line = reader.readLine()) != null) {
            code.add(line);
          }
          String[] codes = code.toArray(new String[code.size()]);
          saveStrings("exported_rf.txt", codes);
          resultArea.setText("Finish exporting! You can find the code in a .txt file at the current path. ");
          
          String errorLine;
          while ((errorLine = errorReader.readLine()) != null) {
              resultArea.append("[ERROR] " + errorLine + "\n");
          }
          errorReader.close();
          
          
          reader.close();
        } 
        catch (IOException e) {
          e.printStackTrace();
        }
    }
    
    if(selectedModel == "XGB"){
      try {
          code.clear();
          Process process = Runtime.getRuntime().exec("/Users/asiu/Desktop/FluxableCode/export_xgb.py");//Add more parameter to transfer them into python document. Starting from sys.argv[1]. 
          BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
    
          BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
          String line;
          while ((line = reader.readLine()) != null) {
            code.add(line);
          }
          String[] codes = code.toArray(new String[code.size()]);
          saveStrings("exported_xgb.txt", codes);
          resultArea.setText("Finish exporting! You can find the code in a .txt file at the current path. ");
          
          String errorLine;
          while ((errorLine = errorReader.readLine()) != null) {
              resultArea.append("[ERROR] " + errorLine + "\n");
          }
          errorReader.close();
          
          
          reader.close();
        } 
        catch (IOException e) {
          e.printStackTrace();
        }
    }
  }
}

void mouseWheel(MouseEvent event) {
    boolean mouseInDisplay = mouseX > 300 && mouseX < width - 10 && mouseY > 0 && mouseY < height - 190;
    if(mouseInDisplay){
      float e = event.getCount();
      zoomFactor += e * 0.1;
      zoomFactor = constrain(zoomFactor, 0.5, 10); 
    }

}

void mouseDragged() {
  boolean mouseInDisplay = mouseX > 300 && mouseX < width - 10 && mouseY > 0 && mouseY < height - 190;
  if(mouseInDisplay){
      if (mouseButton == LEFT) {
          //modelPosition.x = mouseX;
          //modelPosition.y = mouseY;
          modelPosition.x += mouseX - pmouseX; 
          modelPosition.y += mouseY - pmouseY;
      } else if (mouseButton == RIGHT) {
          rotY += (mouseX - pmouseX) * 0.01;
          rotX -= (mouseY - pmouseY) * 0.01;
      }
  }
}

void showRandomItem(){
  if (remainingList.isEmpty()) {
        resultArea.setText("All deformations have been performed!");
        return;
    }
    
    int randomIndex = int(random(remainingList.size()));
    String selectedItem = remainingList.get(randomIndex);
    
    if ("Compressing".equals(selectedItem)) {
        selectedItem += "(o)";
    } else if ("Extending".equals(selectedItem)) {
        selectedItem += "(e)";
    } else if ("Twisting".equals(selectedItem)) {
        selectedItem += "(t)";
    } else if ("Twisting+Extending".equals(selectedItem)) {
        selectedItem += "(r)";
    } else if ("Twisting+Compressing".equals(selectedItem)) {
        selectedItem += "(f)";
    } else if ("Lateral pressing".equals(selectedItem)) {
        selectedItem += "(l)";
    } else if ("Bending".equals(selectedItem)){
        selectedItem += "(b)";
    }

    resultArea.setText("Please do the following movement: " + "\n" + selectedItem + "\n" + "To start, press the hot key and press again to finish. " + "\n" + "Once finished, press the 'Next' button on the right side to move on. ");
    
    remainingList.remove(randomIndex);
}
