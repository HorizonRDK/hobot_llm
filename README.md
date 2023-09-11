# 1. 功能介绍

**hobot_llm** 是地平线RDK平台集成的端侧**Large Language Model (LLM)** Node，用户可在端侧体验LLM。目前提供两种体验方式，一种直接终端输入文本聊天体验，一种订阅文本消息，然后将结果以文本方式发布出去。

# 2. 物料清单

| 机器人名称        | 生产厂家 | 参考链接                                       |
| :---------------- | -------- | ---------------------------------------------- |
| RDK X3（4GB内存） | 多厂家   | [点击跳转](https://developer.horizon.cc/rdkx3) |

# 3. 使用方法

## 3.1. 准备工作

在体验之前，需要具备以下基本条件：

- 确认地平线RDK为4GB内存版本
- 地平线RDK已烧录好地平线提供的Ubuntu 20.04系统镜像
- 安装transformers，命令为 `pip3 install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple`
- 更新hobot-dnn，命令为 `sudo apt update; sudo apt install hobot-dnn`

## 3.2. 安装功能包

启动RDK X3后，通过终端SSH或者VNC连接机器人，复制如下命令在RDK的系统上运行，完成相关Node的安装。

```bash
sudo apt update
sudo apt install -y tros-hobot-llm
```

## 3.3. 运行程序

运行程序前，需要下载模型文件并解压，命令如下：

```bash
# 下载模型文件
wget http://archive.sunrisepi.tech/llm-model/llm_model.tar.gz

# 解压
sudo tar -xf llm_model.tar.gz -C /opt/tros/lib/hobot_llm/
```

同时需要修改BPU保留内存大小为1.7GB，命令如下：

```bash
# 替换dtb文件
sudo cp /opt/tros/lib/hobot_llm/config/hobot-dtb/*.dtb /boot/hobot/

# 重启
reboot
```

重启后调整CPU最高频率为1.5GHz，以及设置调度模式为`performance`，命令如下：

```bash
sudo bash -c 'echo 1 > /sys/devices/system/cpu/cpufreq/boost'
sudo bash -c 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor'
```

目前提供两个运行程序 **hobot_llm_chat** 和 **hobot_llm**，其中 **hobot_llm_chat** 提供终端交互体验，用户可直接输入文本体验大模型，**hobot_llm** 程序订阅 `std_msgs/msg/String` 类型文本消息，送给大模型处理，最后再将结果以 `std_msgs/msg/String` 类型发布出去，该程序可串联其他Node，例如将输出文本语音播放出去。

### 3.3.1. 运行 hobot_llm_chat

```bash
source /opt/tros/setup.bash

ros2 run hobot_llm hobot_llm_chat
```

程序启动后，可直接在当前终端和机器人聊天。

### 3.3.2. 运行 hobot_llm

1. 启动 hobot_llm

    ```bash
    source /opt/tros/setup.bash

    ros2 run hobot_llm hobot_llm
    ```

2. 新开一个终端订阅输出结果topic

    ```bash
    source /opt/tros/setup.bash

    ros2 topic echo /text_result
    ```

3. 新开一个终端发布消息

    ```bash
    source /opt/tros/setup.bash

    ros2 topic pub --once /text_query std_msgs/msg/String "{data: ""中国的首都是哪里""}"
    ```

# 4. 接口说明

**hobot_llm** 程序接口说明如下：

## 4.1. 话题

| 名称        | 消息类型            | 说明              |
| ----------- | ------------------- | ----------------- |
| /text_query | std_msgs/msg/String | 默认订阅topic     |
| /text_result  | std_msgs/msg/String | 默认结果发布topic |

## 4.2. 参数

| 参数名    | 类型        | 解释              | 是否必须 | 支持的配置           | 默认值       |
| --------- | ----------- | ----------------- | -------- | -------------------- | ------------ |
| topic_sub | std::string | 订阅文本topic名称 | 否       | 根据实际部署环境配置 | /text_query  |
| topic_pub | std::string | 发布结果topic名称 | 否       | 根据实际部署环境配置 | /text_result |

# 5. 常见问题

1. 模型加载失败

    确认开发板内存为4GB，同时修改BPU保留内存大小为1.7GB。

2. 输出结果乱码

   确认已使用命令`sudo apt update; sudo apt install hobot-dnn`更新 hobot-dnn。

3. 如何手动修改BPU保留内存为1.7GB？

   修改方法参考[在设备树中设置ion_cam size](https://developer.horizon.ai/api/v1/fileData/documents_rdk/system_software_development/driver_develop_guide/18-Memory_Managment_zh_CN.html#ion-cam-size)，修改 alloc-ranges 和 size 属性中的 0x2a000000 为 0x6a400000。
