PShape myModel;

void setup() {
  size(800, 600, P3D);  // 使用 P3D 渲染器
  background(200);
  
  // 将此处替换为您的模型文件路径
  myModel = loadShape("/Users/asiu/Desktop/SnakeModel_Ready.obj");
}

void draw() {
  background(200);
  
  // 设置摄像机和灯光
  camera(0, -200, height/2.0 / tan(PI*30.0 / 180.0), 0, 0, 0, 0, 1, 0);
  directionalLight(255, 255, 255, 0, -1, -1);
  
  // 将模型移动到画布中心并旋转
  translate(width/2, height/2, 0);
  //rotateY(millis() * 0.001);
  
  // 渲染模型
  scale(20);
  shape(myModel);
}
