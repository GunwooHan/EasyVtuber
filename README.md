# EasyVtuber  

> 用买皮的钱，再买一张3080吧！

![OBS Record With Transparent Virtural Cam Input](assets/new_sample.gif)

Fork自 https://github.com/GunwooHan/EasyVtuber  
为解决面捕质量问题，又反向port了原版demo https://github.com/pkhungurn/talking-head-anime-2-demo 中关于ifacialmocap的ios面捕逻辑  
并且省略了ifacialmocap pc端，通过UDP直连的方式使ios面捕刷新率达到30fps，解决了面捕刷新率的瓶颈  
最后，将EasyVtuber中使用的OBS虚拟摄像头方案切换为Unity Capture，解锁RGBA输出能力，无需绿背即可直接使用

## Requirements  

### 硬件  

- 支持FaceID的iPhone（使用ifacialmocap软件，需购买，需要稳定的WIFI连接）或网络摄像头（使用OpenCV）  
- 支持PyTorch的顶级显卡（参考：TUF RTX3080 默频 26FPS 90%占用）
### 软件

- 本方案在Windows上测试可用
- Anaconda
- OBS或Unity Capture（虚拟摄像头方案）
- 你喜欢的Python IDE（这里使用Pycharm）  
- Photoshop或其他图片处理软件
- 能解决简单报错的脑子和大概半天的折腾时间

## Installation  

### 克隆本Repo  

克隆完以后如果直接用Pycharm打开了，先不要进行Python解释器配置。

### Python和Anaconda环境  

这个项目使用Anaconda进行包管理  
首先前往https://www.anaconda.com/ 安装Anaconda  
启动Anaconda Prompt控制台  
国内用户建议此时切换到清华源（pip和conda都要换掉，尤其是conda的Pytorch Channel，pytorch本体太大了）  
然后运行 `conda env create -f env_conda.yaml` 一键安装所有依赖  
如果有报错（一般是网络问题），删掉配了一半的环境，`conda clean --all`清掉下载缓存，调整配置后再试

安装完成后，在Pycharm内打开本项目，右下角解释器菜单点开，`Add Interpreter...`->`Conda Environment`->`Existing environment`  
选好自己电脑上的`conda.exe`和刚才创建好的`talking-head-anime-2-demo`环境内的`python.exe`    
点击OK，依赖全亮即可  

### 下载预训练模型  

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

### 输入输出设备  

#### UnityCapture  

如果需要使用透明通道输出，参考 https://github.com/schellingb/UnityCapture#installation 安装好UnityCapture  
只需要正常走完Install.bat，在OBS里能看到对应的设备（Unity Video Capture）就行

#### iFacialMocap  

https://www.ifacialmocap.com/download/  
你大概率需要购买正式版（非广告，只是试用版不太够时长）  
购买之前确认好自己的设备支持  
**不需要下载PC软件**，装好iOS端的软件即可，连接信息通过参数传入Python  

## Run

完全体运行命令`python main.py --output_webcam unitycapture --ifm 192.168.31.182:49983 --character y`

参数名 | 值类型 | 说明
:---: | :---: | :---:
--character|字符串|`character`目录下的输入图像文件名，不需要带扩展名
--debug|无|打开OpenCV预览窗口输出渲染结果，如果没有任何输出配置，该参数默认生效
--input|字符串|不使用iOS面捕时，传入要使用的摄像头设备名称，默认为设备0，有ifm参数时无效
--ifm|字符串|使用iOS面捕时，传入设备的`IP:端口号`，如`192.168.31.182:49983`
--output_webcam|字符串|可用值为`obs` `unitycapture`，选择对应的输出种类，不传不输出到摄像头

