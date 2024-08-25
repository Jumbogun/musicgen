### 播放说明
左边乐器栏选择乐器组合，和乐器的渲染音色；下方耳机按钮进行随机生成音乐。

### 运行说明
请安装```./user_interface/requirements.txt```中的```python```依赖库后运行```python main.py```运行播放器

若使用的是conda环境时出现报错
```
OSError: .conda/envs/yourenv/lib/python3.10/site-packages/../../../././libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /usr/lib/libfluidsynth.so.3)
```
可尝试```conda install -c conda-forge libstdcxx-ng```解决


注：PyQt5在版本过高，可能导致界面有瑕疵。另外音乐播放窗口，音频播放的线程管理使用的是PyQt的QThread，目前存在使用中进度条卡顿等问题
