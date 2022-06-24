import numpy as np
import librosa, librosa.display 
import matplotlib.pyplot as plt

FIG_SIZE = (15,10) #시각화 사이즈
sig, sr = librosa.load(file, sr=22050) #음성데이터 가져오기
file = "/content/drive/MyDrive/train/00001.wav" #이 파일에 있는 거

#Waveform
def Waveform(sig):
    plt.figure(figsize=FIG_SIZE)
    librosa.display.waveplot(sig, sr, alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")

#단순 푸리에 변화 -> spectrum
def fft(sig):
    fft = np.fft.fft(sig)

# 복소공간 값 절댓갑 취해서, magnitude 구하기
    magnitude = np.abs(fft) 

# Frequency 값 만들기
    f = np.linspace(0,sr,len(magnitude))

# 푸리에 변환을 통과한 specturm은 대칭구조로 나와서 high frequency 부분 절반을 날려고 앞쪽 절반만 사용한다.
    left_spectrum = magnitude[:int(len(magnitude)/2)]
    left_f = f[:int(len(magnitude)/2)]

    plt.figure(figsize=FIG_SIZE)
    plt.plot(left_f, left_spectrum)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Power spectrum")

#STFT->Spectrogram
def STFT(sig):
# STFT -> spectrogram
    hop_length = 512  # 전체 frame 수
    n_fft = 2048  # frame 하나당 sample 수

# calculate duration hop length and window in seconds
    hop_length_duration = float(hop_length)/sr
    n_fft_duration = float(n_fft)/sr

# STFT
    stft = librosa.stft(sig, n_fft=n_fft, hop_length=hop_length)

# 복소공간 값 절댓값 취하기
    magnitude = np.abs(stft)

# magnitude > Decibels 
    log_spectrogram = librosa.amplitude_to_db(magnitude)

# display spectrogram
    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (dB)")

#mfccs
def mfccs(sig,a):
# MFCCs
# extract 13 MFCCs
    hop_length = 512  # 전체 frame 수
    n_fft = 2048  # frame 하나당 sample 수
    MFCCs = librosa.feature.mfcc(sig, sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=a)

# display MFCCs
    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCC coefficients")
    plt.colorbar()
    plt.title("MFCCs")

# show plots    
    plt.show()

