import speech_recognition as sr
from pydub import AudioSegment
"""
mp3_file = "C:\\Users\\lms110\\Downloads\\record\\장마.mp3"
wav_file = "C:\\Users\\lms110\\Downloads\\record\\장마.wav"

audio = AudioSegment.from_mp3(mp3_file)
audio.export(wav_file, format="wav")
"""

m4a_file = "C:\\Users\\lms110\\Downloads\\record\\example.m4a"
wav_filename = 'example.wav'
track = AudioSegment.from_file(m4a_file, format='m4a')
track.export(wav_filename, format='wav')

recognizer = sr.Recognizer()

m4a_audio = "C:\\Users\\lms110\\Downloads\\record\\example.wav"  # M4A 파일의 경로

output_file_path = "C:\\Users\\lms110\\Downloads\\record\\output.txt"

with sr.AudioFile(wav_filename) as source:
    audio = recognizer.record(source)

try:
    text = recognizer.recognize_google(audio, language='ko-KR')  # 한국어로 변환
    with open(output_file_path, "w", encoding="utf-8") as text_file:
        text_file.write(text)
    print("Text saved to output.txt")
except sr.UnknownValueError:
    print("Google Web Speech API could not understand audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Web Speech API; {e}")

