# 前言

项目上遇到了double类型数据精度问题，嵌入式开发和算法争论了一会有关double和float的精度问题，究竟是强转造成的精度损失更多，还是在计算的过程中精度损失更多？这个问题很显然是使用float在计算过程中造成的精度损失更多，但是面对这样的问题，不能只靠猜测，而是需要进行一系列量化的测算。

# IEEE754标注中的浮点数表达公式

$$ value = (-1)^{sign} \times 2^{exponent} \times (1 + mantissa) $$

其中，sign为符号位，exponent为指数位，mantissa为尾数位。

# float

float类型通常占用4个字节（32位）的内存。具体分配如下：

 - 符号位（Sign bit）：1位

 - 指数位（Exponent）：8位

 - 尾数位（Fraction/Mantissa）：23位

 ## 内存布局示例


假设我们有一个单精度浮点数3.14，它的二进制表示如下：

 - 符号位：0
 - 指数位：10000000
 - 尾数位：10010001111010111000011

 0 10000000 10010001111010111000011

 # double

 - 符号位（Sign bit）：1位
 - 指数位（Exponent）：11位
 - 尾数位（Fraction/Mantissa）：52位

## 内存布局
 假设我们有一个双精度浮点数3.14，它的二进制表示如下：

 - 符号位：0
 - 指数位：10000000000
 - 尾数位：1001000111101011100001010001111010111000010100011110101110000101

# float与doule之间的转换

## float转double

这种转换称为``扩展转换（promotion）``，因为double有更多的位数来表示数字。

### 1.内存模型变化：

 - float使用32位存储，而double使用64位存储。
 - 在将float转换为double时，计算机会将float的值复制到double的尾数部分，并扩展指数部分。
 - 由于double的尾数部分更长，可以精确表示的有效数字更多，所以这种转换通常不会损失精度。

 ### 2.转换过程

 - 符号位保持不变。
 - 指数位从float的8位扩展到double的11位，计算机会根据需要调整指数的偏移量。
 - 尾数位从23位扩展到52位，不足的部分用0填充。

 ## double 转 float

 这种转换称为缩减转换（narrowing），因为float有较少的位数来表示数字。

 ### 1. 内存模型变化：
 -  double使用64位存储，而float使用32位存储。
 - 在将double转换为float时，计算机会将double的值截断或舍入以适应float的尾数部分和指数部分。
 - 由于float的尾数部分较短，这种转换可能会损失精度。


 ### 2.转换过程

 - 符号位保持不变。
 - 指数位从double的11位缩减到float的8位，计算机会调整指数的偏移量，并可能会进行舍入。
 - 尾数位从52位缩减到23位，超出的部分会被截断或舍入，这可能导致精度损失。

 这里需要注意的是，在C++中，并不是做了简单的截断，而是做了舍入操作，这也是为什么我们在实际操作中，可以观测到逢7进1

 # 例子：

## float转换double

假设我们有一个float值3.14：

float:  0 10000000 10010001111010111000011

转换为double，实际上就是把位置往右边填入

double: 0 10000000000 1001000111101011100001010001111010111000010100011110101110000101

可以看到，符号位和尾数位的前23位保持不变，尾数位的其余部分填充为0，指数部分从8位扩展到11位并调整偏移量。

## double转换float

假设我们有一个double值3.14：

double: 0 10000000000 1001000111101011100001010001111010111000010100011110101110000101

转换为float：

float:  0 10000000 10010001111010111000011

符号位保持不变，尾数位截断为前23位，指数部分从11位缩减到8位并调整偏移量。

# 在C++中是如何操作的？

在C++中，double和float的转换是通过编译器实现的。但是我们也可以管中窥豹，看一下具体的实现方式。

## 流程

### 1. 提取 double 的位模式
首先，将 double 类型的值表示为 IEEE 754 双精度浮点数的格式，这包括符号位、指数位和尾数位。

### 2. 分析 double 的位模式

将 double 类型的符号位、指数位和尾数位分别提取出来。对于 double 类型（64位）：

 - 符号位：1位
 - 指数位：11位
 - 尾数位：52位

### 3. 转换指数位

将 double 的指数位转换为 float 的指数位。float 的指数位长度为8位，因此需要进行指数的调整。

 - double 的指数位有11位，偏移量（bias）为1023。
 - float 的指数位有8位，偏移量（bias）为127。

 计算新的指数值：
 $$ new\_exponent = double\_exponent - bias + float\_bias $$

 ### 4.舍入尾数位
 将 double 的尾数位舍入为 float 的尾数位。float 的尾数位长度为23位，而 double 的尾数位长度为52位，因此需要进行舍入操作。

 - 提取 double 尾数位的前23位作为 float 的尾数位。
 - 检查第24位及其后面的位，以确定如何进行舍入。

 ### 5. 组装 float 的位模式
 将符号位、指数位和舍入后的尾数位组装成一个 float 类型的值。

 ## 示例代码

 ``` C++
#include <iostream>
#include <bitset>
#include <cstdint>
#include <cmath>
#include <iomanip>

union DoubleBits {
    double value;
    struct {
        uint64_t mantissa : 52;
        uint64_t exponent : 11;
        uint64_t sign : 1;
    } bits;
};

union FloatBits {
    float value;
    struct {
        uint32_t mantissa : 23;
        uint32_t exponent : 8;
        uint32_t sign : 1;
    } bits;
};

float doubleToFloat(double d) {
    DoubleBits db;
    db.value = d;

    FloatBits fb;
    fb.bits.sign = db.bits.sign;

    // Adjust the exponent
    int32_t new_exponent = db.bits.exponent - 1023 + 127;
    if (new_exponent <= 0) {
        // Underflow
        new_exponent = 0;
        fb.bits.mantissa = 0;
    }
    else if (new_exponent >= 255) {
        // Overflow
        new_exponent = 255;
        fb.bits.mantissa = 0;
    }
    else {
        fb.bits.exponent = new_exponent;
    }

    // Perform rounding on the mantissa
    uint64_t mantissa = db.bits.mantissa;
    uint64_t rounding_mask = 0xFFFFFFFFFF800000; // Mask for the 23 most significant bits
    uint64_t rounding_bits = mantissa & ~rounding_mask;
    uint32_t guard_bit = (mantissa >> 29) & 1;
    uint32_t round_bit = (mantissa >> 28) & 1;
    uint32_t sticky_bit = (mantissa & ((1 << 28) - 1)) != 0;

    mantissa >>= 29;
    if (guard_bit && (round_bit || sticky_bit || (mantissa & 1))) {
        // Round up
        mantissa++;
    }

    fb.bits.mantissa = mantissa & 0x7FFFFF; // Take the 23 least significant bits

    return fb.value;
}

std::string doubleToBinary(double d) {
    DoubleBits db;
    db.value = d;
    std::bitset<64> bits(*reinterpret_cast<uint64_t*>(&d));
    return bits.to_string();
}

std::string floatToBinary(float f) {
    FloatBits fb;
    fb.value = f;
    std::bitset<32> bits(*reinterpret_cast<uint32_t*>(&f));
    return bits.to_string();
}

int main() {
    double d = 3.141592653589793;
    float f = doubleToFloat(d);

    std::cout << std::setprecision(8);
    std::cout << "float: " << f << std::endl;
    std::cout << std::setprecision(16);
    std::cout << "double: " << d << std::endl;


    std::cout << "double (binary): " << doubleToBinary(d) << std::endl;
    std::cout << "float (binary): " << floatToBinary(f) << std::endl;

    return 0;
}


 ```