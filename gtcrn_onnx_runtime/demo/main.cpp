#include "../api/api.h"
#include "../wav_reader/wav_reader.h"
#include "windows.h"
// ���洦������ƵΪWAV�ļ�
bool saveWavFile(const std::string& output_path, const int16_t* data, size_t data_size,
    uint32_t sample_rate, uint16_t num_channels) {
    std::ofstream file(output_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "�޷�������ļ�: " << output_path << std::endl;
        return false;
    }

    // WAV�ļ�ͷ�ṹ
    struct WavHeader {
        char riff[4] = { 'R', 'I', 'F', 'F' };
        uint32_t file_size;
        char wave[4] = { 'W', 'A', 'V', 'E' };

        char fmt[4] = { 'f', 'm', 't', ' ' };
        uint32_t fmt_size = 16;
        uint16_t audio_format = 1;  // PCM
        uint16_t num_channels;
        uint32_t sample_rate;
        uint32_t byte_rate;
        uint16_t block_align;
        uint16_t bits_per_sample = 16;

        char data[4] = { 'd', 'a', 't', 'a' };
        uint32_t data_size;
    };

    WavHeader header;
    header.num_channels = num_channels;
    header.sample_rate = sample_rate;
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.byte_rate = header.sample_rate * header.block_align;
    header.data_size = static_cast<uint32_t>(data_size * sizeof(int16_t));
    header.file_size = 36 + header.data_size;

    // д���ļ�ͷ
    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    // д����Ƶ����
    file.write(reinterpret_cast<const char*>(data), header.data_size);

    if (!file) {
        std::cerr << "WAV�ļ�д��ʧ��" << std::endl;
        return false;
    }

    std::cout << "��������Ƶ�ѱ�����: " << output_path << std::endl;
    return true;
}

int main() {
    // ���ÿ���̨����ΪUTF-8������������ʾ
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    // ��Ƶ�ļ�·��
    const std::string input_audio_path = "D:\\AudioSample\\16khz\\white\\D4_750.wav";
    const std::string output_audio_path = "D:\\AudioSample\\16khz\\white\\D4_750_processed.wav";
    const std::string model_path = "D:\\WorkShop\\Github\\gtcrn_onnx_runtime\\model\\model_trained_on_dns3.onnx";

    // ��Ƶ�������
    const int sample_rate = 16000;  // 16kHz������
    const int frame_size = 512;     // ֡��С
    const int hop_size = 256;       // ��Ծ��С
    const std::string window_type = "hanning";  // ��������

    try {
        // 1. ��ȡ������Ƶ�ļ�
        WavReader wav_reader(input_audio_path);
        if (!wav_reader.isOpen()) {
            std::cerr << "�޷�����Ƶ�ļ�: " << input_audio_path << std::endl;
            return 1;
        }

        // ��ӡ��Ƶ��Ϣ
        std::cout << "������Ƶ��Ϣ:" << std::endl;
        wav_reader.printInfo();

        // ���������Ƿ�Ϊ16000Hz
        if (wav_reader.getSampleRate() != sample_rate) {
            std::cerr << "����: ��Ƶ�����ʱ���Ϊ16000Hz" << std::endl;
            return 1;
        }

        // ��ȡ��Ƶ����
        const std::vector<int16_t>& audio_data = wav_reader.getAudioData();
        std::cout << "��Ƶ������: " << audio_data.size() << std::endl;

        // 2. ��ʼ����Ƶ������
        AudioProcessorContext* processor = audio_processor_init(
            model_path.c_str(),
            sample_rate,
            frame_size,
            hop_size       
        );

        if (!processor) {
            std::cerr << "��Ƶ��������ʼ��ʧ��" << std::endl;
            return 1;
        }

        // 3. ׼��������Ƶ
        std::vector<int16_t> output_data(audio_data.size() * 2);  // Ԥ���㹻��Ļ�����
        size_t output_size = output_data.size();

        // 4. ������Ƶ
        ProcessStatus status = audio_processor_process(
            processor,
            audio_data.data(),
            audio_data.size(),
            output_data.data(),
            &output_size
        );

        if (status != PROCESS_SUCCESS) {
            std::cerr << "��Ƶ����ʧ�ܣ��������: " << status << std::endl;
            audio_processor_destroy(processor);
            return 1;
        }

        // �������������С��ƥ��ʵ�ʴ�����
        output_data.resize(output_size);

        // 5. ���洦������Ƶ
        bool save_success = saveWavFile(
            output_audio_path,
            output_data.data(),
            output_data.size(),
            sample_rate,
            wav_reader.getNumChannels()  // ������������ͬ��������
        );

        if (!save_success) {
            std::cerr << "���洦������Ƶʧ��" << std::endl;
            audio_processor_destroy(processor);
            return 1;
        }

        // 6. ������Դ
        audio_processor_destroy(processor);

        std::cout << "��Ƶ������ɣ�" << std::endl;
        std::cout << "����������: " << audio_data.size() << std::endl;
        std::cout << "���������: " << output_data.size() << std::endl;

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "�����쳣: " << e.what() << std::endl;
        return 1;
    }
}