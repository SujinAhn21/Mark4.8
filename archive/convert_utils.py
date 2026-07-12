# convert_utils.py (디버깅 포함)  # 이제 더이상 .m4a를 안쓰기 때문에 불필요. 
import os
import glob
import shutil
from pydub import AudioSegment

'''colab에는 ffmpeg가 이미 PATH에 있으므로 설정 주석처리'''
# AudioSegment.converter = "C:/ffmpeg-2025-05-05-git-f4e72eb5a3-full_build/bin/ffmpeg.exe"
# AudioSegment.ffprobe   = "C:/ffmpeg-2025-05-05-git-f4e72eb5a3-full_build/bin/ffprobe.exe"
'''colab에서는 gg drive 마운트 후 drive/MyDrive/mark4.1 경로 사용'''
# input_dir="C:/Users/user/Desktop/AI_model/mark4.1/data"
# output_dir="C:/Users/user/Desktop/AI_model/mar4.1/data_wav"

def process_audio_files(input_dir="C:/Users/user/Desktop/AI_model/mark2.5.0/data",
                        output_dir="C:/Users/user/Desktop/AI_model/mark2.5.0/data_wav"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 폴더 '{output_dir}'가 준비되었습니다.")
    print(f"입력 폴더: '{input_dir}'") # 입력 폴더 경로 확인

    m4a_files = glob.glob(os.path.join(input_dir, "*.m4a"))
    wav_files_in_input = glob.glob(os.path.join(input_dir, "*.wav"))

    print(f"'{input_dir}'에서 찾은 m4a 파일 수: {len(m4a_files)}") # m4a 파일 수 확인
    if m4a_files:
        print(f"  첫 번째 m4a 파일 예시: {m4a_files[0]}")
    print(f"'{input_dir}'에서 찾은 wav 파일 수: {len(wav_files_in_input)}") # wav 파일 수 확인
    if wav_files_in_input:
        print(f"  첫 번째 wav 파일 예시: {wav_files_in_input[0]}")


    if not m4a_files and not wav_files_in_input:
        print(f"'{input_dir}' 폴더에 변환하거나 복사할 m4a 또는 wav 파일이 없습니다.")
        return

    converted_count = 0
    copied_count = 0
    
    for m4a_path in m4a_files:
        base = os.path.basename(m4a_path)
        wav_name = os.path.splitext(base)[0] + ".wav"
        wav_path = os.path.join(output_dir, wav_name)
        try:
            audio = AudioSegment.from_file(m4a_path, format='m4a')
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(wav_path, format='wav')
            print(f"m4a 변환 완료: {base} -> {wav_name}")
            converted_count += 1
        except Exception as e:
            print(f"m4a 변환 실패: {base}")
            print(e)

    for wav_path_input in wav_files_in_input:
        base = os.path.basename(wav_path_input)
        output_wav_path = os.path.join(output_dir, base)
        try:
            if os.path.abspath(wav_path_input) == os.path.abspath(output_wav_path):
                print(f"wav 파일 건너뛰기 (입력과 출력이 동일 경로 및 파일): {base}")
                continue
            
            # 복사 전에 출력 파일이 이미 존재하는지 확인 (덮어쓰기 방지 또는 알림)
            if os.path.exists(output_wav_path):
                print(f"wav 파일 복사 건너뛰기 (출력 폴더에 이미 파일 존재): {base} -> {output_wav_path}")
                # 이 경우 copied_count를 증가시키지 않거나, 이미 처리된 것으로 간주할 수 있음
                # 여기서는 이미 존재하면 복사하지 않고 copied_count도 증가시키지 않음
                continue

            shutil.copy2(wav_path_input, output_wav_path)
            print(f"wav 파일 복사 완료: {base} -> {output_wav_path}")
            copied_count += 1
        except shutil.SameFileError:
            print(f"wav 파일 복사 건너뛰기 (소스와 대상이 동일 파일): {base}")
        except Exception as e:
            print(f"wav 파일 복사 실패: {base}")
            print(e)
            
    total_processed = converted_count + copied_count
    print(f"\n총 {len(m4a_files) + len(wav_files_in_input)}개 파일 대상 처리 시도.")
    print(f"  - m4a -> wav 변환: {converted_count}개")
    print(f"  - wav 파일 복사: {copied_count}개")
    print(f"총 {total_processed}개의 파일이 '{output_dir}' 폴더에 저장(또는 이미 존재)되었습니다.")

if __name__ == "__main__":
    process_audio_files()

'''
이 수정된 convert_utils.py를 실행하고 나오는 프린트문을 확인. 
특히 '{input_dir}'에서 찾은 wav 파일 수: 이 부분이 중요. 
만약 0으로 나온다면, data 폴더에 .wav 파일이 없거나 경로 문제일 가능성이 큼큼.
- 가정: 만약 data 폴더에는 .m4a 파일만 있고, data_wav 폴더에 이미 .wav 파일이 일부 존재했다면, 
  generate_dataset_index.py는 data_wav를 기준으로 인덱스를 생성했을 것. 
  이 경우 convert_utils.py의 동작은 .m4a만 변환하는 것이 맞음음. 
만약 data 폴더에도 .wav 파일이 있는데 복사가 안 되었다면 위 디버깅 프린트로 단서 확인.
'''