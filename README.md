# EasyVtuber

![OBS Record With Transparent Virtural Cam Input](assets/new_sample.gif)

Fork自 https://github.com/GunwooHan/EasyVtuber  
为解决面捕质量问题，又反向port了原版demo https://github.com/pkhungurn/talking-head-anime-2-demo 中关于ifacialmocap的ios面捕逻辑  
并且省略了ifacialmocap pc端，通过UDP直连的方式使ios面捕刷新率达到30fps，解决了面捕刷新率的瓶颈  
最后，将EasyVtuber中使用的OBS虚拟摄像头方案切换为Unity Capture，解锁RGBA输出能力，无需绿背即可直接使用

##Requirements  

###硬件  

- 支持FaceID的iPhone（使用ifacialmocap软件，需购买，需要稳定的WIFI连接）或网络摄像头（使用OpenCV）  
- 支持PyTorch的顶级显卡（参考：TUF RTX3080 默频 26FPS 90%占用）
###软件

- 本方案在Windows上测试可用
- Anaconda
- OBS或Unity Capture（虚拟摄像头方案）
- 你喜欢的Python IDE（这里使用Pycharm）  
- Photoshop或其他图片处理软件
- 能解决简单报错的脑子和大概半天的折腾时间

##Installation  

###克隆本Repo  

克隆完以后如果直接用Pycharm打开了，先不要进行Python解释器配置。

###Python和Anaconda环境  

这个项目使用Anaconda进行包管理  
首先前往https://www.anaconda.com/ 安装Anaconda  
启动Anaconda Prompt控制台  
国内用户建议此时切换到清华源（pip和conda都要换掉，尤其是conda的Pytorch Channel，pytorch本体太大了）  
然后运行 `conda env create -f env_conda.yaml` 一键安装所有依赖  
如果有报错（一般是网络问题），就删掉配了一半的环境，`conda clean --all`清掉下载缓存，调整配置后再试

安装完成后，在Pycharm内打开本项目，右下角解释器菜单点开，`Add Interpreter...`  
选好自己电脑上的`conda.exe`和刚才创建好的`talking-head-anime-2-demo`环境内的`python.exe`

###下载预训练模型  

https://github.com/pkhungurn/talking-head-anime-2-demo#download-the-model  
从原repo中下载（this Dropbox link）的压缩文件  
解压到`pretrained`文件夹中，与`PUT_MODEL_HERE`同级  
正确的目录层级为  
```
+ pretrained
  - combiner.pt
  - eyebrow_decomposer.pt
  - eyebrow_morphing_combiner.pt
  - face_morpher.pt
  - two_algo_face_rotator.pt
  - PUT_MODEL_HERE
```