import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def save_normalized_spectrogram(audio_path, save_path):
    # 오디오 파일 로드
    y, sr = librosa.load(audio_path, sr=None)
    
    # 스펙트로그램 생성
    D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
    
    # 1 ~ -1 사이로 정규화
    D_normalized = 2 * (D - np.min(D)) / (np.max(D) - np.min(D)) - 1
    
    # 채널을 3으로 확장 (모든 채널에 동일한 값 사용)
    D_normalized_3ch = np.stack([D_normalized] * 3, axis=-1)  # (224, 224, 3)

    # 정규화된 스펙트로그램 시각화 (그래프 형식 포함)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D_normalized, x_axis='time', y_axis='log', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Normalized Spectrogram')
    
    # 이미지 저장
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def process_audio_folder(audio_folder, output_folder):
    # 출력 폴더가 없으면 생성pip install tensorflow

    os.makedirs(output_folder, exist_ok=True)
    
    # 폴더 내 모든 오디오 파일 처리
    for file_name in os.listdir(audio_folder):
        if file_name.endswith(('.wav', '.mp3', '.flac')):  # 지원하는 오디오 형식
            audio_path = os.path.join(audio_folder, file_name)
            save_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.png")
            save_normalized_spectrogram(audio_path, save_path)
            print(f"Saved: {save_path}")

# 사용 예시
audio_folder = r"C:\Users\Administrator\Desktop\테스트\수정"  # 오디오 파일 폴더 경로
output_folder = r"C:\Users\Administrator\Desktop\테스트\이미지"  # 저장할 이미지 폴더 경로
process_audio_folder(audio_folder, output_folder)
