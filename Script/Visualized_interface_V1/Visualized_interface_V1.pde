import java.awt.GraphicsEnvironment;

void setup() {
  String[] fontNames = GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames();
  for (String name : fontNames) {
    println(name);
  }
  noLoop();
}

void draw() {
  background(220);
}
