// wasapi_capture.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#include <iostream>
#include <mmdeviceapi.h>
#include <Audioclient.h>
#include <conio.h>//_kbhit
#include <stdio.h>
#include <intrin.h>//_byteswap_ulong
#include <mmreg.h>
//-----------------------------------------------------------
// Record an audio stream from the default audio capture
// device. The RecordAudioStream function allocates a shared
// buffer big enough to hold one second of PCM audio data.
// The function uses this buffer to stream data from the
// capture device. The main loop runs every 1/2 second.
//-----------------------------------------------------------

// REFERENCE_TIME time units per second and per millisecond
#define REFTIMES_PER_SEC  10000000
#define REFTIMES_PER_MILLISEC  10000

#define EXIT_ON_ERROR(hres)  \
              if (FAILED(hres)) { goto Exit; }
#define SAFE_RELEASE(punk)  \
              if ((punk) != NULL)  \
                { (punk)->Release(); (punk) = NULL; }

const CLSID CLSID_MMDeviceEnumerator = __uuidof(MMDeviceEnumerator);
const IID IID_IMMDeviceEnumerator = __uuidof(IMMDeviceEnumerator);
const IID IID_IAudioClient = __uuidof(IAudioClient);
const IID IID_IAudioCaptureClient = __uuidof(IAudioCaptureClient);
struct waveheader
{
	//_byteswap_ulong
	DWORD riff = _byteswap_ulong(0x52494646);
	DWORD length;//little endian
	DWORD wave = _byteswap_ulong(0x57415645);
	DWORD fmt = _byteswap_ulong(0x666d7420);
	DWORD filter = (0x00000010);//16 18
	WORD FormatTag;//formattag
	WORD channel;//mono 1 stereo 2
	DWORD samplerate;
	DWORD bytespersec;//samplerate*bitsdepth*channel/8
	WORD samplesize;//bitsdepth*channel/8
	WORD bitsdepth;//32-bit 16-bit 8-bit
	//WORD exinfolength;
	//WORD wvalidbitsdepth;
	//DWORD dwChannelMask;
	//GUID SubFormat;//128b 16B
	DWORD data = _byteswap_ulong(0x64617461);
	DWORD datalength;
};


class MyAudioSink
{
public:
	FILE* pfile = NULL;
	DWORD filelength = 0;
	waveheader mwaveheader;
	BOOL kp = FALSE;
	MyAudioSink(CONST CHAR* filepath)
	{
		pfile = fopen(filepath, "wb+");
	}
	~MyAudioSink()
	{
		//log length&data length
		//offset 4  & 44
		fseek(pfile, 4, 0);
		filelength += -8;
		fwrite(&filelength, 4, 1, pfile);
		fseek(pfile, 36 + 4, 0);//+2+mwaveheader.exinfolength
		filelength += -36;//-2-mwaveheader.exinfolength
		fwrite(&filelength, 4, 1, pfile);

		fclose(pfile);
	}
	HRESULT SetFormat(WAVEFORMATEXTENSIBLE* wfat2)
	{
		WAVEFORMATEX* wfat;
		wfat = (WAVEFORMATEX*)wfat2;
		mwaveheader.FormatTag = WAVE_FORMAT_IEEE_FLOAT;//wfat->wFormatTag
		//FFFE 65534 extensible
		mwaveheader.channel = wfat->nChannels;
		mwaveheader.samplerate = wfat->nSamplesPerSec;
		mwaveheader.bytespersec = wfat->nAvgBytesPerSec;
		mwaveheader.samplesize = wfat->nBlockAlign;
		mwaveheader.bitsdepth = wfat->wBitsPerSample;
		/*mwaveheader.exinfolength= wfat->cbSize;
		mwaveheader.wvalidbitsdepth = wfat2->Samples.wValidBitsPerSample;
		mwaveheader.dwChannelMask = wfat2->dwChannelMask;
		mwaveheader.SubFormat = wfat2->SubFormat;*/
		//"00000003-0000-0010-8000-00aa00389b71", KSDATAFORMAT_SUBTYPE_IEEE_FLOAT)
		filelength = 36 + 8;//+2+mwaveheader.exinfolength
		fwrite(&mwaveheader, filelength, 1, pfile);
		fflush(pfile);
		//write waveheader
		return S_OK;
	}
	HRESULT CopyData(BYTE* pData, UINT32 numFramesAvailable, BOOL* bDone)
	{
		WORD BytesToWrite;
		BytesToWrite = numFramesAvailable * mwaveheader.samplesize;
		WORD offset = 0;

		filelength += BytesToWrite;
		//std::cout << filelength/1024 <<"KB"<< std::endl;
		//std::cout << (filelength-36+8)/mwaveheader.samplerate/mwaveheader.samplesize<<'s'<<std::endl;
		if (pData == NULL)
		{
			BYTE zero[512] = { 0x00 };
			for (int i = 0; i < numFramesAvailable; ++i)
			{
				fwrite(zero, mwaveheader.samplesize, 1, pfile);
			}
		}
		else
		{
			fwrite(pData, mwaveheader.samplesize, numFramesAvailable, pfile);
		}//move buffer data into file(memory)


		if (kp) *bDone = TRUE;
		else kp = _kbhit();

		return S_OK;
	}
};


HRESULT RecordAudioStream(MyAudioSink* pMySink)
{
	HRESULT hr = 0;
	REFERENCE_TIME hnsRequestedDuration = REFTIMES_PER_SEC;
	REFERENCE_TIME hnsActualDuration;
	UINT32 bufferFrameCount;
	UINT32 numFramesAvailable;
	IMMDeviceEnumerator* pEnumerator = NULL;
	IMMDevice* pDevice = NULL;
	IAudioClient* pAudioClient = NULL;
	IAudioCaptureClient* pCaptureClient = NULL;
	WAVEFORMATEX* pwfx = NULL;
	WAVEFORMATEXTENSIBLE* pwfx2 = NULL;
	UINT32 packetLength = 0;
	BOOL bDone = FALSE;
	BYTE* pData;
	DWORD flags;
	CoInitialize(NULL);
	hr = CoCreateInstance(
		CLSID_MMDeviceEnumerator, NULL,
		CLSCTX_ALL, IID_IMMDeviceEnumerator,
		(void**)&pEnumerator);
	EXIT_ON_ERROR(hr)
		//get device(audio endpoint device)
		hr = pEnumerator->GetDefaultAudioEndpoint(
			eRender, eConsole, &pDevice);
	EXIT_ON_ERROR(hr)
		//get audioclient
		hr = pDevice->Activate(
			IID_IAudioClient, CLSCTX_ALL,
			NULL, (void**)&pAudioClient);
	EXIT_ON_ERROR(hr)
		//get format waveformex
		hr = pAudioClient->GetMixFormat(&pwfx);
	pwfx2 = (WAVEFORMATEXTENSIBLE*)pwfx;
	//give it to waveform extensible

	EXIT_ON_ERROR(hr)
		hr = pAudioClient->Initialize(
			AUDCLNT_SHAREMODE_SHARED,
			AUDCLNT_STREAMFLAGS_LOOPBACK,
			hnsRequestedDuration,
			0,
			pwfx,
			NULL);
	EXIT_ON_ERROR(hr)
		// Get the size(audioframes) of the allocated buffer.
		hr = pAudioClient->GetBufferSize(&bufferFrameCount);
	EXIT_ON_ERROR(hr)
		//framesize in bytes=channel * sample size per channel
		//mono(1) stereo(2)
		// stereo 16bit=32b=4B
		hr = pAudioClient->GetService(
			IID_IAudioCaptureClient,
			(void**)&pCaptureClient);
	EXIT_ON_ERROR(hr)
		// Notify the audio sink which format to use.
		hr = pMySink->SetFormat(pwfx2);
	EXIT_ON_ERROR(hr)
		// Calculate the actual duration of the allocated buffer.
		hnsActualDuration = (double)REFTIMES_PER_SEC *
		bufferFrameCount / pwfx->nSamplesPerSec;

	hr = pAudioClient->Start();  // Start recording.
	EXIT_ON_ERROR(hr)
		// Each loop fills about half of the shared buffer.
		while (bDone == FALSE)
		{
			// Sleep for half the buffer duration.
			Sleep(hnsActualDuration / REFTIMES_PER_MILLISEC / 2);
			hr = pCaptureClient->GetNextPacketSize(&packetLength);
			EXIT_ON_ERROR(hr)
				while (packetLength != 0)
				{
					// Get the available data in the shared buffer.
					hr = pCaptureClient->GetBuffer(
						&pData,
						&numFramesAvailable,
						&flags, NULL, NULL);
					//&pData 
					EXIT_ON_ERROR(hr)
						if (flags & AUDCLNT_BUFFERFLAGS_SILENT)
						{
							pData = NULL;  // Tell CopyData to write silence.
						}

					// Copy the available capture data to the audio sink.
					hr = pMySink->CopyData(
						pData, numFramesAvailable, &bDone);
					EXIT_ON_ERROR(hr)
						hr = pCaptureClient->ReleaseBuffer(numFramesAvailable);
					EXIT_ON_ERROR(hr)
						hr = pCaptureClient->GetNextPacketSize(&packetLength);
					EXIT_ON_ERROR(hr)
				}
			bDone = _kbhit();
			std::cout << (pMySink->filelength - 36 + 8) /
				pMySink->mwaveheader.samplerate /
				pMySink->mwaveheader.samplesize
				<< 's';
			std::cout << pMySink->filelength / 1024
				<< "KB" << std::endl;
		}

	hr = pAudioClient->Stop();  // Stop recording.
	EXIT_ON_ERROR(hr)

		Exit:
	CoTaskMemFree(pwfx);
	SAFE_RELEASE(pEnumerator)
		SAFE_RELEASE(pDevice)
		SAFE_RELEASE(pAudioClient)
		SAFE_RELEASE(pCaptureClient)

		return hr;
}


int main()
{
	int i = 1;
	CONST CHAR file[] = "Q:\\workplace\\demo\\wasapi_capture\\wasapi_capture\\2.wav";
	MyAudioSink* psink = new MyAudioSink(file);
	RecordAudioStream(psink);
	delete psink;
	return 0;
}