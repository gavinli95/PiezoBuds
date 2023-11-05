import webrtcvad
import numpy as np


def apply_vad(audio, sr, vad_level=3, frame_duration=30):
    vad = webrtcvad.Vad()
    vad.set_mode(vad_level)

    frame_samples = int(sr * frame_duration / 1000)
    num_frames = len(audio) // frame_samples
    audio_frames = [audio[i:i + frame_samples] for i in range(0, len(audio), frame_samples)]
    audio_frames = audio_frames[:num_frames]

    # Convert the frames to 16-bit PCM format
    audio_frames = [np.round(frame * 32767).astype(np.int16) for frame in audio_frames]

    speech_labels = [vad.is_speech(audio_frames[i].tobytes(), sample_rate=sr) for i in range(num_frames)]
    return speech_labels


def split_audio_to_utterances(audio, sr, speech_labels, frame_duration=30, min_speech_duration=500):
    frame_samples = int(sr * frame_duration / 1000)

    speech_starts = np.where(np.diff(np.array([0] + speech_labels + [0])) == 1)[0]
    speech_ends = np.where(np.diff(np.array([0] + speech_labels + [0])) == -1)[0]

    min_speech_samples = int(sr * min_speech_duration / 1000)

    speech_utterances = []
    for start, end in zip(speech_starts, speech_ends):
        if (end - start) * frame_samples >= min_speech_samples:
            speech_utterances.append((start * frame_samples, end * frame_samples))

    return speech_utterances

