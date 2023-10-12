abstract class UIComponent {
    float x, y, w, h;
    boolean isVisible = true;
    UIComponent(float x, float y, float w, float h) {
        this.x = x;
        this.y = y;
        this.w = w;
        this.h = h;
    }
    abstract void display();
    abstract String getValue();
    abstract void handleMouseClick(float mouseX, float mouseY, SidebarSection section);
}





class SidebarSection {
  float x, y, w, h;
  float contentHeight; 
  color c;
  ArrayList<UIComponent> components = new ArrayList<UIComponent>();
  Dropdown activeDropdown = null;
  float scrollY = 0;
  float maxScrollY;
  
  void addComponent(UIComponent comp) {
        components.add(comp);
  }
  
  SidebarSection(float x, float y, float w, float h, color c) {
    this.x = x;
    this.y = y;
    this.w = w;
    this.h = h;
    this.contentHeight = h;
    this.c = c;
  }

  void display() {
    fill(c); 
    rect(x, y, w, h);
    
    
    contentHeight = 0;
    for (UIComponent comp : components) {
        contentHeight += comp.h + 10; 
    }
    maxScrollY = contentHeight - h;

    pushMatrix();  
    translate(0, -scrollY);
    
    
    for (UIComponent comp : components) {
            if(comp.isVisible){
              comp.display();
            }
    }
    
    
    popMatrix();
    
    if(contentHeight > h) {
        float scrollbarHeight = h * h / contentHeight;
        float scrollbarY = y + scrollY * (h - scrollbarHeight) / maxScrollY;
        fill(150); 
        rect(x + w - 10, scrollbarY, 8, scrollbarHeight);
    }
  }
  
  //Hide all component but the selected one
  void hideAllComponentsExcept(UIComponent exception) {
        for (UIComponent comp : components) {
            if (comp != exception) {
                comp.isVisible = false;
            }
        }
    }
   
   //Show all the components
   void showAllComponents() {
       for (UIComponent comp : components) {
           comp.isVisible = true;
       }
    }
    void mouseScrolled(float count) {
      float scrollAmount = 1; 
      scrollY += count * scrollAmount;
      scrollY = constrain(scrollY, 0, maxScrollY);
    }
}




class Dropdown extends UIComponent {
    String[] options;
    int selectedIndex = 0;
    boolean isOpen = false;
    String label;

    Dropdown(float x, float y, float w, float h, String[] options, String label) {
        super(x, y, w, h);
        this.options = options;
        this.label = label;
    }
    
    public void setX(float x){
      this.x = x;
    }
    
    public void setY(float y){
      this.y = y;
    }

    void display() {
        fill(255);
        rect(x, y, w, h);
        String displayText = isOpen ? options[selectedIndex] : label;
        fill(0);
        text(displayText, x + 10, y + h/2);
        
        if (isOpen) {
            for (int i = 0; i < options.length; i++) {
                float optionY = y + (i+1) * h;
                fill(200); 
                rect(x, optionY, w, h);
                fill(0);
                text(options[i], x + 10, optionY + h/2);
            }
        }
        
    }
    
    void handleMouseClick(float mouseX, float mouseY, SidebarSection section) {
      if (mouseX >= x && mouseX <= x + w && mouseY >= y && mouseY <= y + h) {
        isOpen = !isOpen;
            
        if (isOpen) {
          section.hideAllComponentsExcept(this);
        } else {
          section.showAllComponents();
        }
     }
    }

    String getValue() {
        return options[selectedIndex];
    }
}





class RadioButton extends UIComponent {
    String[] options;
    int selectedIndex = 0;
    
    float circleRadius = 10;

    RadioButton(float x, float y, float w, float h, String[] options) {
        super(x, y, w, h);
        this.options = options;
    }

    void display() {
        for (int i = 0; i < options.length; i++) {
            float circleX = x;
            float circleY = y + i * (2 * circleRadius + 5); // Arranged vertically with each button 5 units apart

            // Drawing the outer circle
            noFill();
            stroke(0);
            ellipse(circleX, circleY, 2 * circleRadius, 2 * circleRadius);

            // If selected, draws an inner filled circle
            if (i == selectedIndex) {
                fill(0);
                ellipse(circleX, circleY, circleRadius, circleRadius);
            }

            // shows the text
            fill(0);
            text(options[i], circleX + 2 * circleRadius, circleY + 5); // Text is moved down slightly to center it vertically
        }
    }
    
    void handleMouseClick(float mouseX, float mouseY, SidebarSection section) {
        for (int i = 0; i < options.length; i++) {
            float circleY = y + i * (2 * circleRadius + 5);
            if (dist(mouseX, mouseY, x, circleY) <= circleRadius) {
                selectedIndex = i;
            }
        }
    }

    String getValue() {
        return options[selectedIndex];
    }
}




















SidebarSection section1, section2, section3, section4;
int fourthHeight;

void setup() {
  size(1200, 800);
  fourthHeight = height / 4;
  section1 = new SidebarSection(0, 0, 150, fourthHeight, #3498DB);
  section2 = new SidebarSection(0, fourthHeight, 150, fourthHeight, #5DADE2);
  section3 = new SidebarSection(0, fourthHeight * 2, 150, fourthHeight, #85C1E9);
  section4 = new SidebarSection(0, fourthHeight * 3, 150, fourthHeight, #AED6F1);
  
  
  section2.addComponent(new Dropdown(section2.x + 35, section2.y + 10, 80, 30, new String[]{"Option1", "Option2", "Option3"}, "Deform_Type"));
  section2.addComponent(new RadioButton(section2.x + 35, section2.y + 100, 80, 30, new String[]{"Radio1", "Radio2"}));
  section2.addComponent(new Dropdown(section2.x + 35, section2.y + 200, 80, 30, new String[]{"OptionA", "OptionB"}, "Param"));
}

void draw() {
  section1.display();
  section2.display();
  section3.display();
  section4.display();
}


void mousePressed() {
    for (UIComponent comp : section2.components) {
        comp.handleMouseClick(mouseX, mouseY, section2);
    }
    //Add more sections here
}

void mouseWheel(MouseEvent event) {
    float count = event.getCount();
    // 判断鼠标是否在 SidebarSection 上
    if (mouseX >= section2.x && mouseX <= section2.x + section2.w) {
        section2.mouseScrolled(count);
    }
    // 为其他 sections 重复相同的操作...
}
