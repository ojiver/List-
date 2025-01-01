# 人工智慧期末報告-在Google Colab 中快速實踐深度學習
<h2>
  組員:  11225035 謝詠任、11225017 黃誠睿
</h2>

# Colaboratory

```Colaboratory```是一個免費的Jupyter 筆記本環境，不需要進行任何設定就可以使用，並且完全在雲端運行。借助Colaboratory，
我們可以在瀏覽器中編寫和執行程式碼、保存和共享分析結果，以及利用強大的運算資源，包含GPU 與TPU 來運行我們的實驗程式碼。


```Colab``` 能夠輕鬆地與Google Driver 與Github 鏈接，我們可以使用[Open in Colab](https://chromewebstore.google.com/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo)  插件快速打開Github 上的Notebook，
或者使用類似於 [https://colab.research.google](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)
...這樣的鏈接打開。如果需要將Notebook 儲存回Github，
直接使用File→Save a copy to GitHub即可。譬如筆者所有與Colab 相關的程式碼歸置在了AIDL-Workbench/colab。


# 依賴與運行時
### 依賴安裝
Colab 提供了便利的依賴安裝功能，允許使用pip 或apt-get 指令進行安裝：

    # Importing a library that is not in Colaboratory
    !pip install -q matplotlib-venn
    !apt-get -qq install -y libfluidsynth1

    # Upgrading TensorFlow
    # To determine which version you're using:
    !pip show tensorflow

    # For the current version:
    !pip install --upgrade tensorflow

    # For a specific version:
    !pip install tensorflow==1.2

    # For the latest nightly build:
    !pip install tf-nightly

    # Install Pytorch
    from os import path
    from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
    platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
    
    accelerator = 'cu80' if path.exists('/opt/bin/nvidia-smi') else 'cpu'
    
    !pip install -q http://download.pytorch.org/whl/{accelerator}/torch-0.4.0-{platform}-linux_x86_64.whl torchvision
    
    # Install 7zip reader libarchive
    # https://pypi.python.org/pypi/libarchive
    !apt-get -qq install -y libarchive-dev && pip install -q -U libarchive
    import libarchive
    
    # Install GraphViz & PyDot
    # https://pypi.python.org/pypi/pydot
    !apt-get -qq install -y graphviz && pip install -q pydot
    import pydot
    
    # Install cartopy
    !apt-get -qq install python-cartopy python3-cartopy
    import cartopy
![](report/a16.jpg)
在Colab 中還可以設定環境變數：

    %env KAGGLE_USERNAME=abcdefgh

![](report/a15.jpg)

# 硬體加速
我們可以透過以下方式查看Colab 為我們提供的硬體：

    from tensorflow.python.client import device_lib
    device_lib.list_local_devices()
    
    !ls /proc
    # CPU信息
    !cat /proc/cpuinfo
    # 内存
    !cat /proc/meminfo
    # 版本
    !cat /proc/version
    # 设备
    !cat /proc/devices
    # 空间
    !df
![](report/a14.jpg)
如果需要為Notebook 啟動GPU 支援：Click Edit->notebook settings->hardware accelerator->GPU，然後在程式碼中判斷是否有可用的GPU 裝置：

    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
      raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
![](report/a13.jpg)
我們可以透過建構經典的CNN 卷積層來比較GPU 與CPU 在運算上的差異：

    import tensorflow as tf
    import timeit
    
    # 確保 GPU 設置正確
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(e)
    
    # 定義操作
    @tf.function
    def cpu():
        random_image_cpu = tf.random.normal((100, 100, 100, 3))
        net_cpu = tf.nn.conv2d(random_image_cpu[tf.newaxis, ...], 
                               filters=tf.random.normal((7, 7, 3, 32)), 
                               strides=1, 
                               padding='SAME')
        return tf.reduce_sum(net_cpu)
    
    @tf.function
    def gpu():
        random_image_gpu = tf.random.normal((100, 100, 100, 3))
        net_gpu = tf.nn.conv2d(random_image_gpu[tf.newaxis, ...], 
                               filters=tf.random.normal((7, 7, 3, 32)), 
                               strides=1, 
                               padding='SAME')
        return tf.reduce_sum(net_gpu)
    
    # 測試執行
    print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
          '(batch x height x width x channel). Sum of ten runs.')
    
    # CPU 測試
    with tf.device('/CPU:0'):
        cpu_time = timeit.timeit(cpu, number=10)
        print(f"CPU (s): {cpu_time}")
    
    # GPU 測試
    with tf.device('/GPU:0'):
        gpu_time = timeit.timeit(gpu, number=10)
        print(f"GPU (s): {gpu_time}")
    
    # 計算 GPU 加速比
    print('GPU speedup over CPU: {:.2f}x'.format(cpu_time / gpu_time))
![](report/a12.jpg)
# 本地運行
Colab 也支援直接將Notebook 連接到本機的Jupyter 伺服器以執行，首先需要啟用jupyter_http_over_ws 擴充功能：

    !pip install jupyter_http_over_ws
    !jupyter serverextension enable --py jupyter_http_over_ws
![](report/a11.jpg)
然後在正常方式啟動Jupyter 伺服器，設定一個標記來明確表明信任來自Colaboratory 前端的WebSocket 連線：

    !jupyter notebook \
      --NotebookApp.allow_origin='https://colab.research.google.com' \
      --port=8888 \
      --NotebookApp.port_retries=0
![](report/a10.jpg)
然後在Colab 的Notebook 中選擇連接到本機程式碼執行程式即可。
# 數據與外部模組
Colab 中的notebook 和py 檔案預設都是以/content/ 作為工作目錄，需要執行指令手動切換工作目錄，例如：

    import os
    path = "/content/drive/MyDrive/113-ai/期末報告"
    os.chdir(path)
    os.listdir(path)
# Google Driver
在過去進行實驗的時候，大量訓練與測試資料的取得、儲存與載入一直是令人頭痛的問題；在Colab 中，
筆者將Awesome DataSets https://url.wx-coder.cn/FqwyP ) 中的相關資料透過AIDL-Workbench/datasets中的腳本持久化儲存在Google Driver 中。
在Colab 中我們可以將Google Driver 掛載到當的工作路徑：

    from google.colab import drive
    drive.mount("/content/drive")
    
    print('Files in Drive:')
    !ls /content/drive/'My Drive'
![](report/a9.jpg)
然後透過正常的Linux Shell 指令來建立與操作：

    # Working with files
    # Create directories for the new project
    !mkdir -p drive/kaggle/talkingdata-adtracking-fraud-detection
    
    !mkdir -p drive/kaggle/talkingdata-adtracking-fraud-detection/input/train
    !mkdir -p drive/kaggle/talkingdata-adtracking-fraud-detection/input/test
    !mkdir -p drive/kaggle/talkingdata-adtracking-fraud-detection/input/valid
    
    # Download files
    !wget -O /content/drive/'My Drive'/Data/fashion_mnist/train-images-idx3-ubyte.gz http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    
    # Download and Unzip files
    %env DIR=/content/drive/My Drive/Data/animals/cats_and_dogs
    
    !rm -rf "$DIR"
    !mkdir -pv "$DIR"
    !wget -O "$DIR"/Cat_Dog_data.zip https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip
    
    # remove existing directories
    !(cd "$DIR" && unzip -qqj Cat_Dog_data.zip -d .)
![](report/a8.jpg)
# 外部Python 文件
Colab 允許我們上傳Python 檔案到工作目錄下，或載入Google Driver 中的Python：

    # Import modules
    import imp
    year = imp.new_module('year')
    exec(open("/content/drive/MyDrive/113-ai/期中報告/year.py").read(), year.__dict__)
    fc_model = imp.new_module('fc_model')
    exec(open("/content/drive/MyDrive/113-ai/期中報告/year.py").read(), fc_model.__dict__)

    
![](report/a7.jpg)


# 文件上傳與下載
Colab 也允許我們在執行腳本時候直接從本機檔案上傳，或將產生的模型下載到本機檔案：

    from google.colab import files
    
    # Upload file
    uploaded = files.upload()
    
    for fn in uploaded.keys():
      print('User uploaded file "{name}" with length {length} bytes'.format(
          name=fn, length=len(uploaded[fn])))
    
    # Download file
    with open('example.txt', 'w') as f:
      f.write('some content')
    
    files.download('example.txt')

![](report/a6.jpg)


![](report/a5.jpg)


# BigQuery
如果我們使用了BigQuery 提供了大數據的查詢與管理功能，那麼在Colab 中也可以直接引入BigQuery 中的資料來源：

    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id)
    
    sample_count = 2000
    row_count = client.query('''
      SELECT
        COUNT(*) as total
      FROM `bigquery-public-data.samples.gsod`''').to_dataframe().total[0]
    
    df = client.query('''
      SELECT
        *
      FROM
        `bigquery-public-data.samples.gsod`
      WHERE RAND() < %d/%d
    ''' % (sample_count, row_count)).to_dataframe()
    
    print('Full dataset has %d rows' % row_count)

# 控制使用
### 網格
Colab 為我們提供了Grid 以及Tab 控件，來便於我們建立簡單的圖表佈局：

    !pip install ipywidgets
    import ipywidgets as widgets
    from IPython.display import display
    import matplotlib.pyplot as plt
    import numpy as np
    
    output1 = widgets.Output()
    output2 = widgets.Output()
    grid = widgets.VBox([widgets.HBox([output1, output2])])
    
    display(grid)
    
    for i in range(20):
        with random.choice([output1, output2]):
            plt.figure(figsize=(2, 2))
            plt.plot(np.random.random(10))
            plt.show()
        time.sleep(0.5)

![](report/a4.jpg)


TabBar 提供了頁籤化的版面：

    from __future__ import print_function
    
    from google.colab import widgets
    from google.colab import output
    from matplotlib import pylab
    from six.moves import zip
    
    
    def create_tab(location):
      tb = widgets.TabBar(['a', 'b'], location=location)
      with tb.output_to('a'):
        pylab.figure(figsize=(3, 3))
        pylab.plot([1, 2, 3])
      # Note you can access tab by its name (if they are unique), or
      # by its index.
      with tb.output_to(1):
        pylab.figure(figsize=(3, 3))
        pylab.plot([3, 2, 3])
        pylab.show()
    
    
    print('Different orientations for tabs')
    
    positions = ['start', 'bottom', 'end', 'top']
    
    for p, _ in zip(positions, widgets.Grid(1, 4)):
      print('---- %s ---' % p)
      create_tab(p)

      
![](report/a3.jpg)


# 表單
值得稱讚的是，Colab 還提供了可互動的表單元件，來方便我們建立可動態輸入的應用：

    #@title String fields
    
    text = 'value' #@param {type:"string"}
    dropdown = '1st option' #@param ["1st option", "2nd option", "3rd option"]
    text_and_dropdown = '2nd option' #@param ["1st option", "2nd option", "3rd option"] {allow-input: true}
    
    print(text)
    print(dropdown)
    print(text_and_dropdown)

    
![](report/a2.jpg)


# 參考資料[在Google Colab 中快速實踐深度學習](https://zhuanlan.zhihu.com/p/69558211)
