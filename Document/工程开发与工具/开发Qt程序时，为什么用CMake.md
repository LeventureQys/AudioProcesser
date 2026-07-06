# 开发Qt程序时，为什么是CMake？

## 什么是CMake？

CMake 是一个跨平台的构建工具，用来管理 C/C++ 项目的编译过程。它通过读取 CMakeLists.txt 配置文件，自动生成适合不同操作系统和编译器的构建脚本（比如 Makefile 或 Visual Studio 项目），让开发者只需写一次配置，就能在各种环境下编译代码。简单说，CMake 帮你省去了手动写复杂编译命令的麻烦。

QMake(由.pro文件组织起来的编译结构)和CMake(由CMakeLists.txt组织起来的编译结构)实际上都是类似的，都是由一个脚本组织整个编译过程，而且都是使用自己的方式简化了整个编译过程。

## 为什么是CMake？

我只说三个原因：

一、Qt Group事实上已经放弃了QMake作为官方的编译工具，仅保留维护更新

二、CMake是大部分C++库编译脚本的事实标准

三、VS对QMake的支持几乎为零，Qt + VS的Qt插件支持，但是这个插件也几乎停止更新了。CMake有VS的支持。

我可以列出一大堆CMake比QMake更优越的理由，但是工程中使用CMake只需要上面三个理由就好了。

## 如何编译CMake的项目

一般情况下，需要在项目文件夹最顶层找到一个CMakeLists.txt，这个编译脚本通常是最顶层的脚本，控制整个编译链。

一个项目里通常由很多个CMakeLists.txt组成的，通常情况是由最上层的一个CMakeLists.txt作为父项目，通过类似add_subdirectory等方法，控制其余多个子项目的编译。

可以使用命令cmake .. 等方式编译，我这里展示使用CMake Gui Tool 编译
![test](https://raw.githubusercontent.com/LeventureQys/Picturebed/main/image/企业微信截图_17506448298314.png)

一般情况，你在上面Where is the source code :中找到一个带有CMakeLists.txt的路径，然后在Where to build the binaries中，把上面的路径copy下来，然后在后面加上一个build文件夹就好了，这样这个项目的所有内容都会编译到这个build文件夹里面来。

如果是第一次编译，可能会弹窗，比如



![test](https://raw.githubusercontent.com/LeventureQys/Picturebed/refs/heads/main/image/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_17506451553032.png)

这里提示是build文件夹不存在是否创建一个，这里点yes创建。


![test](https://raw.githubusercontent.com/LeventureQys/Picturebed/refs/heads/main/image/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_17506453225823.png)

这里第一行是向你询问使用什么生成工具，换句话说就是你想使用哪个IDE对项目进行编译和管理，这里
支持的IDE还挺多的，但是我这里选择使用Visual Studio 2022 编译，谁会拒绝宇宙第一IDE呢？

第二行问的是目标框架，我们由于是在Windows上开发，所以不用选，默认是x64，如果需要交叉编译一些特殊架构的，那你会知道你为什么要选这个，这里我们不选。

第三行问的是有没有一些特殊的参数，不知道的话就是没有

下面四个按钮如果你不知道什么意思就也不用选。

这里点Finish 就可以了。

![test](https://raw.githubusercontent.com/LeventureQys/Picturebed/refs/heads/main/image/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_17506456002724.png)

点击Finish之后，就开始配置CMake项目了，如果参数检查一切正确，那么就会出现Configuring done(xx s)的字样，爆红可能是一些warning而不是error，如果出现Configuring失败，则需要检查具体的error，具体问题具体分析。

比较常见的问题是由于环境变量的问题，上方的变量并没有正确配置，你可以选择手动在CMake Gui Tool 里面手动配置，也可以选择在环境变量里面配置。

这里Configuring Done之后，就可以点Generate了，一般Configure成功Generate不会失败，Generate Done之后，就可以在build文件夹里找到对应的项目管理文件sln了。双击打开，进入VS的CMake项目中。

## 如何在VS中管理CMake项目

生成sln 之后，点开你可能会看到一个ALL_BUILD和一个ZERO_CHECK，这两个项目不用管，相当于是VS为了自己管理CMake而添加的两个默认项目。

1. 如果你编译ALL_BUILD，那么他会自己去编译所有项目

2. ZERO_CHECK的功能是用于​​监控 CMake 配置文件的变更​​并自动重新生成构建系统。


![test](https://raw.githubusercontent.com/LeventureQys/Picturebed/refs/heads/main/image/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_17506459285585.png)

你每个项目内，控制其项目文件的唯一途径是修改CMakeLists.txt，所以请不要像正常使用VS那样将头文件和CPP文件拖动进来。而是先修改CMakeLists.txt，然后再右键它，菜单栏找到编译，VS会自动帮你生成一个新的项目覆盖到原来的，并且会提示你文件已被覆盖是否重新进入项目。

## 常见问题

使用CMake 编译Qt的常见问题主要有两个

### 1. Qt的系统变量配置有问题

CMake编译 Qt需要一个这样的环境变量：

![test](https://raw.githubusercontent.com/LeventureQys/Picturebed/refs/heads/main/image/123123.png)

需要落位到Qt/版本号/编译器类型/lib/cmake/Qt6 文件夹下

### 2. Qt的某个库的路径找不到，CMake的Configure失败

![Test](https://raw.githubusercontent.com/LeventureQys/Picturebed/refs/heads/main/image/123.png)

这里的一些路径可能会出现 NOT-FOUND的情况，暂时不知道是为什么，如果找不到路径，你需要做两件事：

1. 你在其他成功的找到路径的DIR复制下来，把下面名字改一下，假设是这个Qt6LinguistTools_DIR的路径找不到了，你从上面复制一个D:/Devtools/Qt/6.8.0/msvc2022_64/lib/cmake/Qt6Gui下来，把这个Qt6Gui改成Qt6LinguistTools试试先。

2. 如果再次Configure仍然NOT-FOUND，需要确认这个工具是否存在，比如这里有一些和QMultiMedia有关的依赖，但是你的Qt并没有安装QMultiMedia的组件，这个可能会导致这个问题。





