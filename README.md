# EasyVtuber

![OBS Record With Transparent Virtural Cam Input](assets/new_sample.gif)

Fork自 https://github.com/GunwooHan/EasyVtuber  
为解决面捕质量问题，又反向port了原版demo https://github.com/pkhungurn/talking-head-anime-2-demo 中关于ifacialmocap的ios面捕逻辑  
并且省略了ifacialmocap pc端，通过UDP直连的方式使ios面捕刷新率达到30fps，解决了面捕刷新率的瓶颈  
最后，将EasyVtuber中使用的OBS虚拟摄像头方案切换为Unity Capture，解锁RGBA输出能力，无需绿背即可直接使用