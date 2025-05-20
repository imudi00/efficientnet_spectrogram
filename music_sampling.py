import librosa
import soundfile as sf
import os
import numpy as np

# 오디오 파일들이 있는 디렉터리 경로
audio_directory = r"C:\Users\Administrator\Desktop\Sample\01.원천데이터"  # 경로 수정
save_directory = r"C:\Users\Administrator\Desktop\테스트\수정"  # 자른 음악 파일을 저장할 경로

# 디렉터리에 있는 모든 오디오 파일 가져오기 (확장자가 .wav, .mp3 등인 파일들)
audio_files = [f for f in os.listdir(audio_directory) if f.endswith(('.wav', '.mp3', '.flac'))]

# 100개 이상의 음악 파일을 처리
for audio_file in audio_files:
    audio_path = os.path.join(audio_directory, audio_file)
    y, sr = librosa.load(audio_path, sr=16000)

    # 🔥 "비트 변화가 큰 지점" 찾기 (backtrack=True 사용)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)

    # onset_frames는 프레임 인덱스 배열이므로 이를 시간으로 변환
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)

    # onset_strength 값을 onset_frames에 맞춰 다시 계산
    onset_strength = onset_strength[onset_frames]

    # 가장 큰 변화가 일어난 지점 찾기
    max_onset_idx = np.argmax(onset_strength)  # 가장 큰 변화가 일어난 인덱스
    max_onset_time = onset_times[max_onset_idx]  # 그 지점의 시간

    # 해당 지점에서 1분씩 자르기
    segment_length = 60  # 1분
    start_time = max_onset_time  # 가장 큰 비트 변화 지점

    # 전체 음악 길이를 구하고, 자른 구간이 60초 미만이면 뒤에서부터 60초 자르기
    end_time = librosa.get_duration(y=y, sr=sr)
    
    if end_time - start_time < segment_length:
        # 끝에서부터 60초를 잘라야 할 경우
        start_time = max(0, end_time - segment_length)  # 끝에서부터 60초
        end_time = min(start_time + segment_length, librosa.get_duration(y=y, sr=sr))
    else:
        end_time = start_time + 60
    
    # 🎵 "자른 오디오" 저장
    os.makedirs(save_directory, exist_ok=True)

    # 원본 파일 이름을 그대로 사용
    file_name, file_extension = os.path.splitext(audio_file)  # 확장자 분리
    segment_filename = os.path.join(save_directory, f"{file_name}{file_extension}")  # 원본 이름 + "_sliced"

    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    segment = y[start_sample:end_sample]

    sf.write(segment_filename, segment, sr)
    print(f"✅ {segment_filename} 저장 완료! (구간: {start_time:.2f}초 ~ {end_time:.2f}초)")
