#include "CalculateCore.h"
CalculateCore* CalculateCore::m_instance = nullptr;
// 对数插值函数
template <typename Type>
Type logInterpolation(Type x, Type x0, Type x1, Type y0, Type y1) {
	return y0 + (y1 - y0) * (std::log10(x) - std::log10(x0)) / (std::log10(x1) - std::log10(x0));
}

// 线性插值函数
template <typename Type>
Type linearInterpolation(Type x, Type x0, Type x1, Type y0, Type y1) {
	return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

// 横轴上每个点对应频率的值
const QMap<double, double> map_freq_core = {
	{0.00, 1}, {1.00, 20.0}, {2.00, 50.0}, {3.00 , 100.0}, {4.00 , 200.0},
	{5.00, 500.0}, {6.00, 1000.0}, {7.00, 2000.0}, {8.00, 5000.0},
	{9.00, 10000.0}, {10.00, 20000.0}, {11.00, 50000.0}
};

double Actual_to_View(double actual_freq) {
	if (actual_freq < 0) {
		throw std::invalid_argument("Frequency must be non-negative.");
	}

	if (actual_freq == 0) {
		return 0.0;
	}

	auto itLow = map_freq_core.constBegin() + 1;
	auto itHigh = map_freq_core.constEnd();
	--itHigh;

	// 如果实际频率小于第一个点，直接返回第一个点的值
	if (actual_freq <= itLow.value()) {
		return linearInterpolation(actual_freq, 0.0, itLow.value(), 0.0, itLow.key());
		return 0.0;
	}

	// 二分查找合适的区间
	while (std::distance(itLow, itHigh) > 1) {
		auto itMid = itLow;
		std::advance(itMid, std::distance(itLow, itHigh) / 2);

		if (itMid.value() < actual_freq) {
			itLow = itMid;
		}
		else {
			itHigh = itMid;
		}
	}

	double db_low = itLow.value();
	double db_high = itHigh.value();
	double key_low = itLow.key();
	double key_high = itHigh.key();

	// 在合适的区间内进行对数插值
	double view_coord = logInterpolation(actual_freq, db_low, db_high, key_low, key_high);

	// 防止超出边界
	if (view_coord < 0.0) view_coord = 0.0;
	if (view_coord > 11.0) view_coord = 11.0;

	return view_coord;
}


// Round a number to a specified number of decimal places
double roundToDecimalPlaces(double value, int decimalPlaces) {
	double scale = pow(10.0, decimalPlaces);
	return std::round(value * scale) / scale;
}
void printVector(const std::vector<double>& vec, const std::string& label) {
	std::cout << label << ":" << std::endl;
	for (double val : vec) {
		std::cout << val << " ";
	}
	std::cout << std::endl;
}
CalculateCore::CalculateCore()
{
	this->InitCalculatCore();
}

CalculateCore* CalculateCore::ins()
{
	if (!m_instance) {
		m_instance = new CalculateCore();
	}
	return m_instance;
}

CalculateCore::~CalculateCore()
{
}

void CalculateCore::SetMinMaxFrequency(double min, double max)
{
	this->min = min;
	this->max = max;
}

void CalculateCore::InitCalculatCore()
{
	this->default_frequency = { 31.36,49.53,79.74,126,199,314,496,799,1230,1990,3150,4970,8000,12400,19900 };
	this->vec_freq_positions.resize(446);
	for (int i = 0; i < vec_freq_positions.size(); ++i) {
		double ret = min * std::pow((max / min), double(i) / double(steps - 1));
		this->vec_freq_positions[i] = ret;
		if (ret < 100.0) {
			this->vec_freq_positions[i] = roundToDecimalPlaces(ret, 1);
		}
		else if (ret >= 100.0 && ret < 1000.0) {
			this->vec_freq_positions[i] = std::round(ret);
		}
		else if (ret >= 1000.0 && ret < 10000.0) {
			this->vec_freq_positions[i] = roundToDecimalPlaces(ret, -1);
		}
		else if (ret > 10000.0) {
			this->vec_freq_positions[i] = roundToDecimalPlaces(ret, -2);
		}
		//if (ret < 100.0) {
		//	this->vec_freq_10_100.push_back(roundToDecimalPlaces(ret, 1));
		//}
		//else if (ret >= 100.0 && ret < 1000.0) {
		//	this->vec_freq_100_1000.push_back(std::round(ret));
		//}
		//else if (ret >= 1000.0 && ret < 10000.0) {
		//	this->vec_freq_1000_10000.push_back(roundToDecimalPlaces(ret, -1));
		//}
		//else if (ret > 10000.0) {
		//	this->vec_freq_above_10000.push_back(roundToDecimalPlaces(ret, -2));
		//}
	}

	this->vec_Q_positions.resize(q_steps);
	for (int n = 0; n < q_steps; ++n) {
		double ret = qmin * std::pow((qmax / qmin), double(n) / double(q_steps - 1));
		//this->vec_Q_positions[n] = ret;
		if (qmax == 128.8) {
			if (ret < 1.5) {
				this->vec_Q_positions[n] = roundToDecimalPlaces(ret, 2);
			}
			else if (ret >= 1.5 && ret < 15.0) {
				this->vec_Q_positions[n] = roundToDecimalPlaces(ret, 1);
			}
			else if (ret >= 15) {
				this->vec_Q_positions[n] = std::round(ret);
			}
		}
		else {
			if (ret < 1.5) {
				this->vec_Q_positions[n] = roundToDecimalPlaces(ret, 3);
			}
			else if (ret >= 1.5 && ret < 15.0) {
				this->vec_Q_positions[n] = roundToDecimalPlaces(ret, 2);
			}
			else if (ret >= 15) {
				this->vec_Q_positions[n] = roundToDecimalPlaces(ret, 1);
			}
		}
		
		//if (ret < 1.5) {
		//	this->vec_Q_04_15.push_back(roundToDecimalPlaces(ret, 2));
		//}
		//else if (ret >= 1.5 && ret < 15.0) {
		//	this->vec_Q_15_150.push_back(roundToDecimalPlaces(ret, 1));
		//}
		//else if (ret >= 15) {
		//	this->vec_Q_above_150.push_back(std::round(ret));
		//}
	}
}

double CalculateCore::FindNearAvaliableFrequency(double input) {
	//找到最近的输入项
	int index = static_cast<uint32_t>(std::round(std::log(input / min) / std::log(max / min) * double(steps - 1)));

	return this->vec_freq_positions[index];
}

double CalculateCore::FindNearAvaliableFrequency(double input, qint8 offset)
{
	//找到最近的输入项
	int index = static_cast<uint32_t>(std::round(std::log(input / min) / std::log(max / min) * double(steps - 1)));

	//如果index + offset的范围合理
	if (index + offset >= 0 && index + offset < vec_freq_positions.size() - 1) {
		//需要判断当前这个是否向着我们合理的位置去寻找了
		return this->vec_freq_positions[index + offset];
	}
	if (index + offset >= 0 && index + offset < vec_freq_positions.size() - 1) {
		return this->vec_freq_positions[index + offset];
	}
	else if (index + offset < 0) {
		return this->vec_freq_positions[0];
	}
	else if (index + offset > vec_freq_positions.size() - 1) {
		return this->vec_freq_positions[this->vec_freq_positions.size() - 1];
	}
	return this->vec_freq_positions[index];
}

double CalculateCore::FindNearAvaliableQValue(double input, BandType type)
{
	return this->FindNearAvaliableQValue(input, 0, type);
}

double CalculateCore::FindNearAvaliableQValue(double input, qint8 offset, BandType type)
{
	bool blnRemakeQPositions = false;
	if (type == BandType::P) {
		this->qmax = 128.0;
		if (*this->vec_Q_positions.rbegin() < 127.0) {
			blnRemakeQPositions = true;
		}
	}
	else {
		this->qmax = 1.6;
		if (*this->vec_Q_positions.rbegin() > 1.65) {
			//如果是这样的话则需要重新计算这个vec的值了
			blnRemakeQPositions = true;
		}
	}

	if (blnRemakeQPositions) {
		this->vec_Q_positions.resize(q_steps);
		for (int n = 0; n < q_steps; ++n) {
			double ret = qmin * std::pow((qmax / qmin), double(n) / double(q_steps - 1));
			//this->vec_Q_positions[n] = ret;
			if (qmax == 128.0) {
				if (ret < 1.5) {
					this->vec_Q_positions[n] = roundToDecimalPlaces(ret, 2);
				}
				else if (ret >= 1.5 && ret < 15.0) {
					this->vec_Q_positions[n] = roundToDecimalPlaces(ret, 1);
				}
				else if (ret >= 15) {
					this->vec_Q_positions[n] = std::round(ret);
				}
			}
			else {
				if (ret < 1.5) {
					this->vec_Q_positions[n] = roundToDecimalPlaces(ret, 3);
				}
				else if (ret >= 1.5 && ret < 15.0) {
					this->vec_Q_positions[n] = roundToDecimalPlaces(ret, 2);
				}
				else if (ret >= 15) {
					this->vec_Q_positions[n] = roundToDecimalPlaces(ret, 1);
				}
			}
			

			//if (ret < 1.5) {
			//	this->vec_Q_04_15.push_back(roundToDecimalPlaces(ret, 2));
			//}
			//else if (ret >= 1.5 && ret < 15.0) {
			//	this->vec_Q_15_150.push_back(roundToDecimalPlaces(ret, 1));
			//}
			//else if (ret >= 15) {
			//	this->vec_Q_above_150.push_back(std::round(ret));
			//}
		}
	}


	int index = int(std::round(std::log(input / qmin) / std::log(qmax / qmin) * double(q_steps - 1)));
	if (index + offset >= 0 && index + offset < vec_Q_positions.size() - 1) {
		return this->vec_Q_positions[index + offset];
	}
	else if (index + offset < 0) {
		return this->vec_Q_positions[0];
	}
	else if (index + offset >= vec_Q_positions.size()) {
		return this->vec_Q_positions[this->vec_Q_positions.size() - 1];
	}
	return this->vec_Q_positions[index + offset];
}

std::vector<double> CalculateCore::GetDefaultFrequencies()
{
	return default_frequency;
}
#include "qdebug.h"
TFZ_coefficients CalculateCore::MakeCoeffPeakFilter(double center_freq, double QValue, double Gain) {
	TFZ_coefficients result;
	double A = std::pow(10, (Gain / 40));
	double omega = 2.0 * M_PI * center_freq / (double)PublicVar::ins().sample_rate;
	if (false) {
		qDebug() << (double)PublicVar::ins().sample_rate;
	}
	double alpha = 0.5 * std::sin(omega) / QValue;

	double c = -2.0 * std::cos(omega);
	double alphaTimesA = alpha * A;
	double alphaOverA = alpha / A;

	result.b0 = 1.0 + alphaTimesA;
	result.b1 = c;
	result.b2 = 1.0 - alphaTimesA;
	double a0 = 1.0 + alphaOverA;
	result.a1 = c;
	result.a2 = 1.0 - alphaOverA;

	result.b0 /= a0;
	result.b1 /= a0;
	result.b2 /= a0;
	result.a1 /= a0;
	result.a2 /= a0;
	return result;
}

QMap<TFZType, TFZ_coefficients> CalculateCore::MakeCoeffPeakFilterList(double center_freq, double QValue, double Gain)
{
	QMap<TFZType, TFZ_coefficients> map_ret;


	QList<TFZType> list_type = { TFZType::k44_1,TFZType::k48,TFZType::k96,TFZType::k192 };
	for (auto item : list_type) {
		TFZ_coefficients result;
		double A = std::pow(10, (Gain / 40));
		double omega = 2.0 * M_PI * center_freq / (double)item;
		if (false) {
			qDebug() << (double)item;
		}
		double alpha = 0.5 * std::sin(omega) / QValue;

		double c = -2.0 * std::cos(omega);
		double alphaTimesA = alpha * A;
		double alphaOverA = alpha / A;

		result.b0 = 1.0 + alphaTimesA;
		result.b1 = c;
		result.b2 = 1.0 - alphaTimesA;
		double a0 = 1.0 + alphaOverA;
		result.a1 = c;
		result.a2 = 1.0 - alphaOverA;

		result.b0 /= a0;
		result.b1 /= a0;
		result.b2 /= a0;
		result.a1 /= a0;
		result.a2 /= a0;
		map_ret.insert(item, result);
	}
	return map_ret;
}

TFZ_coefficients CalculateCore::MakeCoeffLowShelf(double center_freq, double QValue, double Gain)
{

	TFZ_coefficients ret;
	double A = std::pow(10.0, (Gain / 40.0));
	double aminus1 = A - 1.0;
	double aplus1 = A + 1.0;
	double omega = 2.0 * M_PI * center_freq / (double)PublicVar::ins().sample_rate;
	double coso = std::cos(omega);
	double beta = std::sin(omega) * sqrt(A) / QValue;

	double aminus1TimesCoso = aminus1 * coso;
	double aplus1TimesCoso = aplus1 * coso;
	double a0 = aplus1 + aminus1TimesCoso + beta;
	ret.b0 = (A * (aplus1 - aminus1TimesCoso + beta)) / a0;;
	ret.b1 = (2.0 * A * (aminus1 - aplus1TimesCoso)) / a0;
	ret.b2 = (A * (aplus1 - aminus1TimesCoso - beta)) / a0;

	ret.a1 = (-2.0 * (aminus1 + aplus1TimesCoso)) / a0;
	ret.a2 = (aplus1 + aminus1TimesCoso - beta) / a0;

	return ret;
}

QMap<TFZType, TFZ_coefficients> CalculateCore::MakeCoeffLowShelfList(double center_freq, double QValue, double Gain)
{

	QMap<TFZType, TFZ_coefficients> map_ret;


	QList<TFZType> list_type = { TFZType::k44_1,TFZType::k48,TFZType::k96,TFZType::k192 };
	for (auto item : list_type) {
		TFZ_coefficients ret;
		double A = std::pow(10.0, (Gain / 40.0));
		double aminus1 = A - 1.0;
		double aplus1 = A + 1.0;
		double omega = 2.0 * M_PI * center_freq / (double)item;
		double coso = std::cos(omega);
		double beta = std::sin(omega) * sqrt(A) / QValue;

		double aminus1TimesCoso = aminus1 * coso;
		double aplus1TimesCoso = aplus1 * coso;
		double a0 = aplus1 + aminus1TimesCoso + beta;
		ret.b0 = (A * (aplus1 - aminus1TimesCoso + beta)) / a0;;
		ret.b1 = (2.0 * A * (aminus1 - aplus1TimesCoso)) / a0;
		ret.b2 = (A * (aplus1 - aminus1TimesCoso - beta)) / a0;

		ret.a1 = (-2.0 * (aminus1 + aplus1TimesCoso)) / a0;
		ret.a2 = (aplus1 + aminus1TimesCoso - beta) / a0;
		map_ret.insert(item, ret);
	}
	return map_ret;
}

TFZ_coefficients CalculateCore::MakeCoeffHighShelf(double center_freq, double QValue, double Gain)
{


	TFZ_coefficients ret;
	double A = std::pow(10.0, (Gain / 40.0));
	double aminus1 = A - 1.0;
	double aplus1 = A + 1.0;
	double omega = 2.0 * M_PI * center_freq / (double)PublicVar::ins().sample_rate;
	double coso = std::cos(omega);
	double beta = sin(omega) * sqrt(A) / QValue;
	double aminus1TimesCoso = aminus1 * coso;
	double aplus1TimesCoso = aplus1 * coso;

	double a0 = aplus1 - aminus1TimesCoso + beta;
	ret.b0 = (A * (aplus1 + aminus1TimesCoso + beta)) / a0;;
	ret.b1 = (-2.0 * A * (aminus1 + aplus1TimesCoso)) / a0;
	ret.b2 = (A * (aplus1 + aminus1TimesCoso - beta)) / a0;

	ret.a1 = (2.0 * (aminus1 - aplus1TimesCoso)) / a0;
	ret.a2 = (aplus1 - aminus1TimesCoso - beta) / a0;

	return ret;
}

QMap<TFZType, TFZ_coefficients> CalculateCore::MakeCoeffHighShelfList(double center_freq, double QValue, double Gain)
{

	QMap<TFZType, TFZ_coefficients> map_ret;


	QList<TFZType> list_type = { TFZType::k44_1,TFZType::k48,TFZType::k96,TFZType::k192 };
	for (auto item : list_type) {
		TFZ_coefficients ret;
		double A = std::pow(10.0, (Gain / 40.0));
		double aminus1 = A - 1.0;
		double aplus1 = A + 1.0;
		double omega = 2.0 * M_PI * center_freq / (double)item;
		double coso = std::cos(omega);
		double beta = sin(omega) * sqrt(A) / QValue;
		double aminus1TimesCoso = aminus1 * coso;
		double aplus1TimesCoso = aplus1 * coso;

		double a0 = aplus1 - aminus1TimesCoso + beta;
		ret.b0 = (A * (aplus1 + aminus1TimesCoso + beta)) / a0;;
		ret.b1 = (-2.0 * A * (aminus1 + aplus1TimesCoso)) / a0;
		ret.b2 = (A * (aplus1 + aminus1TimesCoso - beta)) / a0;

		ret.a1 = (2.0 * (aminus1 - aplus1TimesCoso)) / a0;
		ret.a2 = (aplus1 - aminus1TimesCoso - beta) / a0;
		map_ret.insert(item, ret);
	}
	return map_ret;
}

TFZ_coefficients CalculateCore::MakeCoeffBand(BandType type, double center_freq, double QValue, double Gain)
{
	switch (type) {
	case BandType::P: {
		return this->MakeCoeffPeakFilter(center_freq, QValue, Gain);
	}case BandType::LS: {
		return this->MakeCoeffLowShelf(center_freq, QValue, Gain);
	}case BandType::HS: {
		return this->MakeCoeffHighShelf(center_freq, QValue, Gain);
	}
	}
	return TFZ_coefficients();
}

QMap<TFZType, TFZ_coefficients> CalculateCore::MakeCoeffBandList(BandType type, double center_freq, double QValue, double Gain,bool blnBypass)
{
	QList<TFZType> list_type = { TFZType::k44_1,TFZType::k48,TFZType::k96,TFZType::k192 };
	QMap<TFZType, TFZ_coefficients> map_rets;
	for (auto item : list_type) {
		TFZ_coefficients ret;
		double Gain_input = blnBypass ? 0: Gain;
		switch (type) {
		case BandType::P: {
			return this->MakeCoeffPeakFilterList(center_freq, QValue, Gain_input);
		}case BandType::LS: {
			return this->MakeCoeffLowShelfList(center_freq, QValue, Gain_input);
		}case BandType::HS: {
			return this->MakeCoeffHighShelfList(center_freq, QValue, Gain_input);
		}
		}
	}
	return map_rets;
}

TFZ_coefficients CalculateCore::MakeCoeffLowPass(double center_freq)
{
	TFZ_coefficients ret;
	double C = 1 / std::tan(M_PI * center_freq / (double)PublicVar::ins().sample_rate);
	double SqC = std::pow(C, 2);
	double MultC = sqrt(2) * C;
	double c = 1.0 / (1.0 + MultC + SqC);

	ret.b0 = c;
	ret.b1 = 2.0 * c;
	ret.b2 = c;
	ret.a1 = 2.0 * c * (1.0 - SqC);
	ret.a2 = c * (1.0 - MultC + SqC);
	return ret;
}

QMap<TFZType, TFZ_coefficients> CalculateCore::MakeCoeffLowPassList(double center_freq, bool blnBypass)
{
	double input_freq = center_freq;
	QMap<TFZType, TFZ_coefficients> map_ret;
	QList<TFZType> list_type = { TFZType::k44_1,TFZType::k48,TFZType::k96,TFZType::k192 };
	for (auto item : list_type) {
		//if (blnBypass) {
		//	if (item == TFZType::k44_1 || item == TFZType::k48) {
		//		input_freq = 22000;
		//	}
		//	else if (item == TFZType::k96 || item == TFZType::k192) {
		//		input_freq = 48000;
		//	}
		//}
		TFZ_coefficients ret;
		double C = 1 / std::tan(M_PI * input_freq / (double)item);
		double SqC = std::pow(C, 2);
		double MultC = sqrt(2) * C;
		double c = 1.0 / (1.0 + MultC + SqC);

		ret.b0 = c;
		ret.b1 = 2.0 * c;
		ret.b2 = c;
		ret.a1 = 2.0 * c * (1.0 - SqC);
		ret.a2 = c * (1.0 - MultC + SqC);

		ret.blnBypass = blnBypass;
		map_ret.insert(item, ret);
	}
	return map_ret;
}

TFZ_coefficients CalculateCore::MakeCoeffHighPass(double center_freq)
{
	TFZ_coefficients ret;
	double C = std::tan(M_PI * center_freq / (double)PublicVar::ins().sample_rate);
	double SqC = std::pow(C, 2);
	double MultC = sqrt(2) * C;
	double c = 1.0 / (1.0 + MultC + SqC);
	ret.b0 = c;
	ret.b1 = -2.0 * c;
	ret.b2 = c;
	ret.a1 = 2.0 * c * (SqC - 1.0);
	ret.a2 = c * (1.0 - MultC + SqC);
	return ret;
}

QMap<TFZType, TFZ_coefficients> CalculateCore::MakeCoeffHighPassList(double center_freq,bool blnBypass)
{
	QMap<TFZType, TFZ_coefficients> map_ret;


	QList<TFZType> list_type = { TFZType::k44_1,TFZType::k48,TFZType::k96,TFZType::k192 };
	for (auto item : list_type) {
		TFZ_coefficients ret;
		double C = std::tan(M_PI * center_freq / (double)item);
		double SqC = std::pow(C, 2);
		double MultC = sqrt(2) * C;
		double c = 1.0 / (1.0 + MultC + SqC);
		ret.b0 = c;
		ret.b1 = -2.0 * c;
		ret.b2 = c;
		ret.a1 = 2.0 * c * (SqC - 1.0);
		ret.a2 = c * (1.0 - MultC + SqC);
		ret.blnBypass = blnBypass;
		map_ret.insert(item, ret);
	}
	return map_ret;
}

std::map<double, double> CalculateCore::FreqResponse(const TFZ_coefficients& coeffs, int n, bool bypass)
{
	std::map<double, double> response;
	double f_start = 1.0;
	double f_end = (double)PublicVar::ins().nyquist_pattern;
	double step = pow((f_end / f_start), 1.0 / (double(n) - 1.0));

	for (int i = 0; i < n; ++i) {
		double f = f_start * pow(step, i);
		if (bypass) {
			response[f] = 0.0;
		}
		else {
			double w = 2.0 * M_PI * f / (double)PublicVar::ins().sample_rate;
			double pha = 4.0 * pow(sin(w / 2.0), 2);
			double numerator = pow((coeffs.b0 + coeffs.b1 + coeffs.b2), 2) + pha * (coeffs.b0 * coeffs.b2 * pha - (coeffs.b1 * (coeffs.b0 + coeffs.b2) + 4.0 * coeffs.b0 * coeffs.b2));
			double denominator = pow((1.0 + coeffs.a1 + coeffs.a2), 2) + pha * (coeffs.a2 * pha - (coeffs.a1 * (1.0 + coeffs.a2) + 4.0 * coeffs.a2));
			response[f] = 10 * log10(numerator / denominator);
		}
	}
	return response;
}

std::map<double, double> CalculateCore::FreqResponse(const Point& point, int n, bool bypass)
{
	return this->FreqResponse(MakeCoeffBand(point.type, point.frequence, point.Q, point.Gain), n, bypass);
	return std::map<double, double>();
}

std::map<double, double> CalculateCore::TotalFreqResponse(const std::vector<TFZ_coefficients>& input, const TFZ_coefficients& LPF, const TFZ_coefficients& HPF, int n, bool bypass)
{
	std::map<double, double> map_ret;
	std::map<double, double> lpf_response, hpf_response;

	if (!bypass) {
		lpf_response = FreqResponse(LPF, n, false);
		hpf_response = FreqResponse(HPF, n, false);
	}

	if (input.size() == 0) {
		if (!bypass) {
			// 如果输入为空，但未绕过滤波器，则只返回滤波器响应
			for (const auto& item : lpf_response) {
				map_ret[item.first] += item.second;
			}
			for (const auto& item : hpf_response) {
				map_ret[item.first] += item.second;
			}
		}
		return map_ret;
	}
	for (const auto& item : input) {
		std::map<double, double> map_temp = FreqResponse(item, n, false);
		if (map_ret.empty()) {
			map_ret = map_temp;
		}
		else {
			for (auto& entry : map_temp) {
				map_ret[entry.first] += entry.second;
			}
		}
	}
	if (!bypass) {
		// 将LPF和HPF的响应添加到总响应中
		for (const auto& item : lpf_response) {
			map_ret[item.first] += item.second;
		}
		for (const auto& item : hpf_response) {
			map_ret[item.first] += item.second;
		}
	}
	return map_ret;
}

std::map<double, double> CalculateCore::TotalFreqResponse(const std::vector<Point>& vec_points, double LPF_Freq, double HPF_Freq, int n, bool bypass)
{
	std::vector< TFZ_coefficients> vec_coeff;
	for (const auto& item : vec_points) {
		TFZ_coefficients coeff = this->MakeCoeffBand(item.type, item.frequence, item.Q, item.Gain);
		vec_coeff.push_back(coeff);
	}
	TFZ_coefficients LPF = this->MakeCoeffLowPass(LPF_Freq);
	TFZ_coefficients HPF = this->MakeCoeffHighPass(HPF_Freq);

	return TotalFreqResponse(vec_coeff, LPF, HPF, n, bypass);
}
#include "qjsondocument.h"
#include "qjsonobject.h"
void CalculateCore::QFreqResponse(const TFZ_coefficients& input, QVector<double>& frequs, QVector<double>& Gain, int n, bool bypass)
{
	frequs.clear();
	Gain.clear();
	double f_start = 1.0;
	bool blnCheck = false;
	double f_end = (double)PublicVar::ins().nyquist_pattern;
	double step = pow((f_end / f_start), 1.0 / (double(n) - 1.0));
	for (int i = 0; i < n; ++i) {
		double f = f_start * pow(step, i);
		double gain_temp = 0.0;
		if (bypass) gain_temp = 0.0;
		else {
			double w = 2.0 * M_PI * f / (double)PublicVar::ins().sample_rate;
			double pha = 4.0 * pow(sin(w / 2.0), 2);
			double numerator = pow((input.b0 + input.b1 + input.b2), 2) + pha * (input.b0 * input.b2 * pha - (input.b1 * (input.b0 + input.b2) + 4.0 * input.b0 * input.b2));
			double denominator = pow((1.0 + input.a1 + input.a2), 2) + pha * (input.a2 * pha - (input.a1 * (1.0 + input.a2) + 4.0 * input.a2));
			gain_temp = 10 * log10(numerator / denominator);
		}

		frequs.push_back(Actual_to_View(f));
		//frequs.push_back(f);
		Gain.push_back(gain_temp);


	}
	return;
}



void CalculateCore::QTotalFreqResponse(const QVector<Point>& vec_points, QVector<double>& input_freq, QVector<double>& input_Gain, double LPF_Freq, double HPF_Freq, int n, bool bypass)
{
	//只需要算一组freq的值，但是每个freq的点都需要算15 + 2 = 17组gain值进行相加
	std::map<double, double> map_lpf;
	std::map<double, double> map_hpf;
	if (LPF_Freq == -1.0) {
		map_lpf = this->FreqResponse(this->MakeCoeffLowPass(LPF_Freq), n, true);
	}
	else if (LPF_Freq == -1.0) {
		map_lpf = this->FreqResponse(this->MakeCoeffLowPass(LPF_Freq), n, true);
	}

	double f_start = 1.0;
	double f_end = (double)PublicVar::ins().nyquist_pattern;
	double step = pow((f_end / f_start), 1.0 / (double(n) - 1.0));
	//vec_freq只插入一次
	input_freq.clear();
	input_Gain.clear();
	input_freq.resize(n);
	input_Gain.resize(n);

	for (int i = 0; i < n; ++i) {
		input_freq[i] = Actual_to_View(f_start * pow(step, i));
		double f_start = 1.0;
		double f_end = (double)PublicVar::ins().nyquist_pattern;
		double step = pow((f_end / f_start), 1.0 / (double(n) - 1.0));
		double f = f_start * pow(step, i);

		if (bypass) {
			input_Gain[i] = 0.0;
			continue;
		}
		else {
			//计算这个频点上十五个点的效应叠起来的效果
			for (const auto& point : vec_points) {
				//计算一下TFZ参数
				TFZ_coefficients input = this->MakeCoeffBand(point.type, point.frequence, point.Q, point.Gain);
				double w = 2.0 * M_PI * f / (double)PublicVar::ins().sample_rate;
				double pha = 4.0 * pow(sin(w / 2.0), 2);
				double numerator = pow((input.b0 + input.b1 + input.b2), 2) + pha * (input.b0 * input.b2 * pha - (input.b1 * (input.b0 + input.b2) + 4.0 * input.b0 * input.b2));
				double denominator = pow((1.0 + input.a1 + input.a2), 2) + pha * (input.a2 * pha - (input.a1 * (1.0 + input.a2) + 4.0 * input.a2));
				//除了加点本身的Gain值，还需要加上LPF和HPF的值
				input_Gain[i] = input_Gain[i] + 10 * log10(numerator / denominator);
			}
			//计算这个频点上LPF和HPF的效果和
			input_Gain[i] = input_Gain[i] + map_lpf[f] + map_hpf[f];
		}
	}
}