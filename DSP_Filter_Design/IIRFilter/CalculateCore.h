#pragma once
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <map>
#include "DataClass.h"
#include <QVector>
#include "PublicVar.h"
/// @brief Z传递函数中的系数组 Transfer Funtion Coefficients
struct TFZ_coefficients {
	/// @brief b0, b1, b2 are the coefficients of numerator in z transfer function`
	double b0;
	double b1;
	double b2;
	/// @brief a1, a2 are the coefficients of denominator in z transfer function
	double a1;
	double a2;

	//不想这么做的，但是很多地方不好传了
	//需要注意的是，这个值只用于输出，只在输出时修改，请不要随意调用此参数 
	bool blnBypass = false;
};

enum class TFZType {
	k44_1 = 44100,
	k48 = 48000,
	k96 = 96000,
	k192=192000
};

/**

	@class   CalculateCore
	@brief   Calculate Core of Equalizer Curves
	@details ~

**/
class CalculateCore
{
public:

	CalculateCore();
	static CalculateCore* ins();
	~CalculateCore();

	/// <summary>
	/// 初始化运算核心
	/// </summary>
	void InitCalculatCore();

	//void SetSampleRate(double rate);
	double GetSampleRate();

	/// <summary>
	/// 找到最靠近的频率值
	/// </summary>
	/// <param name="input">input </param>
	/// <returns>nearest frequency</returns>
	double FindNearAvaliableFrequency(double input);
	/// <summary>
	/// 找到最靠近的频率值
	/// </summary>
	/// <param name="input">input</param>
	/// <param name="offset">结果值的偏移量</param>
	/// <returns></returns>
	double FindNearAvaliableFrequency(double input, qint8 offset);
	/// <summary>
	/// 找到最靠近的Q值
	/// </summary>
	/// <param name="input">input</param>
	/// <returns> nearest q value</returns>
	double FindNearAvaliableQValue(double input,BandType type = BandType::P);

	/// <summary>
	/// 找到最靠近的Q值
	/// </summary>
	/// <param name="input">input</param>
	/// <param name="offset">offset</param>
	/// <returns>nearest q value</returns>
	double FindNearAvaliableQValue(double input, qint8 offset, BandType type = BandType::P);
	/// <summary>
	/// 获得一组长度为15的默认频点
	/// </summary>
	/// <returns>get default frequencies on equlizer view</returns>
	std::vector<double> GetDefaultFrequencies();

	/// <summary>
	/// second-order low-pass Butterworth filter
	/// </summary>
	/// <param name="center_freq">center_freq cut-off frequency</param>
	/// <returns>Transfer Funtion Coefficients</returns>
	TFZ_coefficients MakeCoeffLowPass(double center_freq);
	QMap<TFZType, TFZ_coefficients> MakeCoeffLowPassList(double freq_center,bool blnBypass = false);
	/// <summary>
	/// peaking/low shelf/high shelf filter
	/// </summary>
	/// <param name="type">filter type</param>
	/// <param name="center_freq">peaking center/shelf midpoint frequency</param>
	/// <param name="QValue">quality factor</param>
	/// <param name="Gain">scale factor (dB)</param>
	/// <returns>Transfer Funtion Coefficients</returns>
	TFZ_coefficients MakeCoeffBand(BandType type, double center_freq, double QValue, double Gain);
	QMap<TFZType,TFZ_coefficients> MakeCoeffBandList(BandType type, double center_freq, double QValue, double Gain,bool blnBypass = false);
	/// <summary>MakeCoeffLowPass
	/// high-pass Butterworth filter
	/// </summary>
	/// <param name="center_freq">cut-off frequency</param>
	/// <returns>Transfer Funtion Coefficients</returns>
	TFZ_coefficients MakeCoeffHighPass(double center_freq);
	QMap<TFZType, TFZ_coefficients> MakeCoeffHighPassList(double freq_center,bool blnBypass = false);
	/// <summary>
	/// frequency response of second-order IIR filter
	/// </summary>
	/// <param name="input">Transfer Funtion Coefficients</param>
	/// <param name="n">point count</param>
	/// <param name="bypass">bypass</param>
	/// <returns> points position</returns>
	std::map<double, double> FreqResponse(const TFZ_coefficients& input, int n, bool bypass = false);

	/// <summary>
	/// frequency response of second-order IIR filter
	/// </summary>
	/// <param name="type">band type P LS HS?</param>
	/// <param name="freq">center frequency of the band</param>
	/// <param name="gain"></param>
	/// <param name="Q"></param>
	/// <param name="n">point on view</param>
	/// <param name="bypass">bypass</param>
	/// <returns></returns>
	std::map<double, double> FreqResponse(const Point& point, int n, bool bypass = false);

	/// <summary>
	/// calculate total freq response lines
	/// </summary>
	/// <param name="input">list of all Transfer Funtion Coefficients </param>
	/// <param name="n">point count</param>
	/// <param name="bypass">bypass</param>
	/// <returns> points position</returns>
	std::map<double, double> TotalFreqResponse(const std::vector<TFZ_coefficients>& input, const TFZ_coefficients& LPF, const TFZ_coefficients& HPF, int n, bool bypass = false);

	std::map<double, double> TotalFreqResponse(const std::vector<Point>& vec_points, double LPF_Freq, double HPF_Freq, int n, bool bypass = false);

	/// <summary>
	/// 绘图专用,返回点属性
	/// </summary>
	/// <param name="vec_points"></param>
	/// <param name="input_freq"></param>
	/// <param name="input_Gain"></param>
	/// <param name="LPF_Freq"></param>
	/// <param name="HPF_Freq"></param>
	/// <param name="n"></param>
	/// <param name="bypass"></param>
	void QTotalFreqResponse(const QVector<Point>& vec_points, QVector<double>& input_freq, QVector<double>& input_Gain, double LPF_Freq, double HPF_Freq, int n, bool bypass = false);

	/// <summary>
	/// 返回Qt的响应曲线
	/// </summary>
	/// <param name="input">Transfer Funtion Coefficients</param>
	/// <param name="n">point count</param>
	/// <param name="bypass">bypass</param>
	/// <returns> points position</returns>
	void QFreqResponse(const TFZ_coefficients& input, QVector<double>& frequs, QVector<double>& Gain, int n, bool bypass = false);

	//Get Different Coefficients from different sample rate



private:

	//默认采样率
	//double current_sample_rate = 44100;

	double min = 10.0; //hz
	double max = 48000.0;  //hz
	int steps = 446;

	void SetMinMaxFrequency(double min, double max);

	std::vector<double> default_frequency;
	std::vector<double> vec_freq_positions;
	std::vector<double> vec_freq_10_100;
	std::vector<double> vec_freq_100_1000;
	std::vector<double> vec_freq_1000_10000;
	std::vector<double> vec_freq_above_10000;

	double qmin = 0.4;
	double qmax = 128.0;
	int q_steps = 102;

	std::vector<double> vec_Q_positions;
	std::vector<double> vec_Q_04_15;
	std::vector<double> vec_Q_15_150;
	std::vector<double> vec_Q_above_150;

	/// <summary>
/// second-order peaking/low shelf/high shelf filter
/// </summary>
/// <param name="center_freq">center_freq peaking center/shelf midpoint frequency</param>
/// <param name="QValue">QValue quality factor</param>
/// <param name="Gain">Gain scale factor</param>
/// <returns>Transfer Funtion Coefficients</returns>
	TFZ_coefficients MakeCoeffPeakFilter(double center_freq, double QValue, double Gain);
	QMap<TFZType, TFZ_coefficients> MakeCoeffPeakFilterList(double center_freq, double QValue, double Gain);
	/// <summary>
	/// low shelf filter
	/// </summary>
	/// <param name="center_freq">shelf midpoint frequency</param>
	/// <param name="QValue">quality factor</param>
	/// <param name="Gain">scale factor that the low frequencies are multiplied by</param>
	/// <returns>Transfer Funtion Coefficients</returns>
	TFZ_coefficients MakeCoeffLowShelf(double center_freq, double QValue, double Gain);
	QMap<TFZType, TFZ_coefficients> MakeCoeffLowShelfList(double center_freq, double QValue, double Gain);
	/// <summary>
	/// high shelf filter for treble control
	/// </summary>
	/// <param name="center_freq">shelf midpoint frequency</param>
	/// <param name="QValue">quality factor</param>
	/// <param name="Gain">scale factor that the low frequencies are multiplied by</param>
	/// <returns>Transfer Funtion Coefficients</returns>
	TFZ_coefficients MakeCoeffHighShelf(double center_freq, double QValue, double Gain);
	QMap<TFZType, TFZ_coefficients> MakeCoeffHighShelfList(double center_freq, double QValue, double Gain);
private:
	static CalculateCore* m_instance; // 静态成员变量，用于存储唯一实例
};
